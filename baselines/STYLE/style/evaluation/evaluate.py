"""
Evaluation script for comparing STYLE implementation with paper metrics.
"""

import torch
import json
from typing import Dict, List, Any
from ..models.disp import DISP
from ..models.retriever import Retriever
from ..utils.llm_integration import LLMIntegration
from ..utils.monitoring import Monitor
from .metrics import Metrics
from ..config import Config
import argparse
import torch.serialization
from ..data.dataset_manager_full import DatasetManager
from torch.utils.data import DataLoader
from dtaidistance import dtw


class Evaluator:
    def __init__(
        self, disp: DISP, retriever: Retriever, llm: LLMIntegration, monitor: Monitor
    ):
        """
        Initialize the evaluator.

        Args:
            disp: Trained DISP model
            retriever: Document retriever
            llm: LLM integration
            monitor: Monitoring utility
        """
        self.disp = disp
        self.retriever = retriever
        self.llm = llm
        self.monitor = monitor
        self.metrics = Metrics()

    def _prepare_state(self, batch):
        """Prepare state tensor from batch."""
        try:
            # Get last query and document
            last_query = (
                batch["query_history"][-1] if batch.get("query_history") else ""
            )
            last_doc = batch["documents"][0] if batch.get("documents") else ""

            # Handle retrieval scores properly
            if isinstance(batch.get("retrieval_scores"), torch.Tensor):
                if batch["retrieval_scores"].numel() > 0:
                    # Take mean of scores if multiple exist
                    last_score = batch["retrieval_scores"][0].mean().item()
                else:
                    last_score = 0.0
            else:
                last_score = (
                    batch["retrieval_scores"][0]
                    if batch.get("retrieval_scores")
                    else 0.0
                )

            # Tokenize inputs
            query_tokens = self.disp.bert_tokenizer(
                last_query,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.disp.device)

            doc_tokens = self.disp.bert_tokenizer(
                last_doc,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.disp.device)

            # Get BERT embeddings
            with torch.no_grad():
                query_outputs = self.disp.bert_encoder(**query_tokens)
                doc_outputs = self.disp.bert_encoder(**doc_tokens)

                # Get [CLS] token embeddings
                query_embedding = query_outputs.last_hidden_state[
                    :, 0, :
                ]  # [1, bert_dim]
                doc_embedding = doc_outputs.last_hidden_state[:, 0, :]  # [1, bert_dim]

            # Convert score to tensor
            score_tensor = torch.tensor([[last_score]], device=self.disp.device)

            # Concatenate features
            state = torch.cat([query_embedding, doc_embedding, score_tensor], dim=1)

            return state

        except Exception as e:
            print(f"Error preparing state: {e}")
            # Create default state tensor with correct dimension
            default_dim = self.disp.dqn_input_dim
            return torch.zeros((1, default_dim), device=self.disp.device)

    def _simulate_ask(self, state):
        """Simulate asking a question."""
        try:
            # Get last query from history
            last_query = state["query_history"][-1] if state["query_history"] else ""

            # Generate question using LLM
            response = self.llm.simulate_user_response(
                user_intent=last_query, question="What would you like to know?"
            )
            return response
        except Exception as e:
            print(f"Error in ask simulation: {e}")
            return "Could you tell me more?"

    def _simulate_clarify(self, state):
        """Simulate clarifying a question."""
        try:
            # Get last query and documents
            last_query = state["query_history"][-1] if state["query_history"] else ""
            docs = state["documents"]

            if not docs:
                raise ValueError("No documents available for clarification")

            # Generate clarification using LLM
            response = self.llm.simulate_user_response(
                user_intent=last_query, question=f"Could you clarify about {docs[0]}?"
            )
            return response
        except Exception as e:
            print(f"Error in clarify simulation: {e}")
            return "I need more information."

    def _simulate_answer(self, state):
        """Simulate answering a question."""
        try:
            # Get last query and documents
            last_query = state["query_history"][-1] if state["query_history"] else ""
            docs = state["documents"]

            if not docs:
                raise ValueError("No documents available for answer")

            # Generate answer using LLM
            response = self.llm.simulate_user_response(
                user_intent=last_query, question=f"Here's what I found: {docs[0]}"
            )
            return response
        except Exception as e:
            print(f"Error in answer simulation: {e}")
            return "I couldn't find a good answer."

    def _update_state(self, state_dict, action, response):
        """Update state based on action and response."""
        # Create a new state dictionary with proper copying
        new_state = {
            "query_history": state_dict["query_history"] + [response],  # Append to list
            "documents": state_dict["documents"],  # Keep reference
            "retrieval_scores": state_dict["retrieval_scores"],  # Keep reference
            "state_tensor": state_dict["state_tensor"].clone(),  # Clone tensor
        }

        # Update documents if needed
        if action == Config.ACTION_ANSWER:
            try:
                new_docs = self.retriever.retrieve(response)
                new_state["documents"] = new_docs
                # Update state tensor with new document
                new_state["state_tensor"] = self._prepare_state(
                    {
                        "query_history": new_state["query_history"],
                        "documents": new_state["documents"],
                        "retrieval_scores": new_state["retrieval_scores"],
                    }
                )
            except Exception as e:
                print(f"Error updating documents: {e}")

        return new_state

    def _check_done(self, state):
        """Check if episode is done."""
        # Check if max turns reached
        if len(state.get("query_history", [])) >= Config.MAX_TURNS:
            return True

        # Check if we have a good answer
        if state.get("documents") and len(state["documents"]) > 0:
            return True

        return False

    def _check_success(self, state_dict):
        """Check if the episode was successful."""
        try:
            # Get target rank
            target_rank = self._get_target_rank(state_dict)

            # Check if target is in top 5
            if target_rank == -1:
                print("Warning: Invalid target rank")
                return False

            return target_rank < 5
        except Exception as e:
            print(f"Error checking success: {e}")
            return False

    def _get_target_rank(self, state_dict):
        """Get the rank of the target document."""
        try:
            if not state_dict.get("documents"):
                return -1

            # Get target document from state
            target_doc = state_dict.get("target_document")
            if not target_doc:
                print("Warning: No target document found in state")
                return -1

            # Find target document in retrieved documents
            for i, doc in enumerate(state_dict["documents"]):
                if doc == target_doc:
                    return i

            print(f"Warning: Target document not found in retrieved documents")
            return -1  # Target not found
        except Exception as e:
            print(f"Error getting target rank: {e}")
            return -1

    def evaluate_domain(self, domain: str, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a specific domain."""
        self.disp.eval()
        episode_results = []

        with torch.no_grad():
            for batch in test_loader:
                # Verify target document exists
                if not batch.get("target_document"):
                    print(f"Warning: No target document in batch")
                    continue

                # Initialize state dictionary
                state_dict = {
                    "query_history": batch.get("query_history", []),
                    "documents": batch.get("documents", []),
                    "retrieval_scores": batch.get("retrieval_scores", []),
                    "target_document": batch["target_document"],
                    "state_tensor": self._prepare_state(batch),
                }

                # Verify documents exist
                if not state_dict["documents"]:
                    print(f"Warning: No documents in batch")
                    continue

                done = False
                turn = 0
                action_sequence = []
                target_rank = -1
                ranks_before = []
                ranks_after = []

                while not done and turn < Config.MAX_TURNS:
                    # Get initial target rank
                    if turn == 0:
                        target_rank = self._get_target_rank(state_dict)
                        ranks_before.append(target_rank)

                    # Select action
                    try:
                        action = self.disp.select_action(state_dict["state_tensor"])
                    except Exception as e:
                        print(f"Error selecting action: {e}")
                        action = 0  # Default to ask

                    action_sequence.append(action)

                    # Execute action
                    if action == Config.ACTION_ASK:
                        response = self._simulate_ask(state_dict)
                    else:
                        response = self._simulate_answer(state_dict)
                        target_rank = self._get_target_rank(state_dict)

                    # Update state
                    state_dict = self._update_state(state_dict, action, response)

                    # Check if done
                    done = self._check_done(state_dict)
                    if done:
                        success = self._check_success(state_dict)
                        print(
                            f"Episode ended: success={success}, turns={turn+1}, target_rank={target_rank}"
                        )

                    # Store transition
                    episode_results.append(
                        {
                            "state": state_dict["state_tensor"],
                            "action": action,
                            "next_state": state_dict["state_tensor"],
                            "done": done,
                            "num_turns": turn + 1,
                            "success": self._check_success(state_dict),
                            "target_rank": target_rank,
                            "action_sequence": action_sequence.copy(),
                            "ranks_before": ranks_before.copy(),
                            "ranks_after": ranks_after.copy(),
                        }
                    )

                    turn += 1

        # Compute metrics
        metrics = self.metrics.compute_all_metrics(episode_results)

        # Log results
        self.monitor.log_metrics(
            {
                f"{domain}/recall@5": metrics["recall@5"],
                f"{domain}/sr@3": metrics["sr@3"],
                f"{domain}/sr@5": metrics["sr@5"],
                f"{domain}/avg_turns": metrics["avg_turns"],
            }
        )

        return metrics

    def _run_episode(
        self, domain: str, user_intent: str, target_doc: str, max_turns: int
    ) -> Dict[str, Any]:
        """
        Run a single evaluation episode.

        Args:
            domain: Domain name
            user_intent: User's true intent
            target_doc: Target document
            max_turns: Maximum number of turns

        Returns:
            Episode results
        """
        # Initialize episode state
        query_history = []
        state = None
        num_turns = 0
        action_sequence = []
        ranks_before = []
        ranks_after = []

        while num_turns < max_turns:
            # Get user query (simulated)
            if not query_history:
                try:
                    query = self.llm.simulate_user_response(
                        user_intent, "What would you like to know?"
                    )
                except Exception as e:
                    print(f"Error generating user response: {e}")
                    # Fallback to using user intent directly
                    query = user_intent
            else:
                query = query_history[-1]

            # Retrieve documents
            try:
                retrieved_docs, retrieval_scores = self.retriever.retrieve(
                    query, domain=domain, top_k=Config.TOP_K_DOCS
                )
            except Exception as e:
                print(f"Error in retrieval: {e}")
                # Fallback to empty results
                retrieved_docs = []
                retrieval_scores = []

            # Get target rank before clarification
            target_rank_before = self._get_target_rank(retrieved_docs)
            ranks_before.append(target_rank_before)

            # Construct state
            try:
                state = self.disp.construct_features(
                    query_history, retrieved_docs, retrieval_scores
                )
            except Exception as e:
                print(f"Error constructing state: {e}")
                # Fallback to empty state
                state = torch.zeros(
                    (1, Config.state_input_dim()), device=Config.DEVICE
                )

            # Select action
            try:
                # Remove epsilon parameter for evaluation
                action, _ = self.disp.select_action(state)
            except Exception as e:
                print(f"Error selecting action: {e}")
                # Fallback to showing results
                action = 0
            action_sequence.append(action)

            if action == Config.ACTION_ASK:
                try:
                    # Generate clarification question
                    question = self.llm.generate_clarification_question(
                        query_history, retrieved_docs, retrieval_scores
                    )

                    # Get user response
                    response = self.llm.simulate_user_response(user_intent, question)
                    query_history.append(response)

                    # Retrieve documents again after clarification
                    retrieved_docs, retrieval_scores = self.retriever.retrieve(
                        response, domain=domain, top_k=Config.TOP_K_DOCS
                    )
                except Exception as e:
                    print(f"Error in clarification: {e}")
                    # Fallback to showing results
                    action = 0
                    retrieved_docs = []
                    retrieval_scores = []

                # Get target rank after clarification
                target_rank_after = self._get_target_rank(retrieved_docs)
                ranks_after.append(target_rank_after)

            else:  # Show results
                target_rank_after = target_rank_before
                ranks_after.append(target_rank_after)

            # Check if target found
            if target_rank_after < 5:
                break

            num_turns += 1

        return {
            "success": target_rank_after < 5,
            "num_turns": num_turns,
            "target_rank": target_rank_after,
            "action_sequence": action_sequence,
            "ranks_before": ranks_before,
            "ranks_after": ranks_after,
        }

    def compare_with_paper(self, results: Dict[str, Dict[str, float]]):
        """
        Compare evaluation results with paper metrics.

        Args:
            results: Dictionary of domain results
        """
        paper_metrics = {
            "ClariQ": {
                "recall@5": 0.6387,
                "sr@3": 0.7647,
                "sr@5": 0.8655,
                "avg_turns": 3.8403,
            },
            "OpenDialKG": {
                "recall@5": 0.4956,
                "sr@3": 0.6144,
                "sr@5": 0.6511,
                "avg_turns": 5.5678,
            },
        }

        comparison = {}
        for domain, metrics in results.items():
            if domain in paper_metrics:
                comparison[domain] = {}
                for metric, value in metrics.items():
                    if metric in paper_metrics[domain]:
                        comparison[domain][metric] = {
                            "our_result": value,
                            "paper_result": paper_metrics[domain][metric],
                            "difference": value - paper_metrics[domain][metric],
                        }
                    else:
                        print(
                            f"Warning: Metric {metric} not found in paper metrics for {domain}"
                        )

        # Log comparison
        self.monitor.log_metrics(
            {
                f"comparison/{domain}/{metric}": diff["difference"]
                for domain, metrics in comparison.items()
                for metric, diff in metrics.items()
            }
        )

        return comparison

    def compute_dtw_distance(self, seq1, seq2):
        """Compute DTW distance between two sequences."""
        try:
            # Move tensors to CPU and convert to numpy
            seq1_np = seq1.cpu().numpy()
            seq2_np = seq2.cpu().numpy()

            # Compute DTW distance
            distance = dtw(seq1_np, seq2_np)
            return distance
        except Exception as e:
            print(f"Warning: DTW computation failed: {e}")
            return float("inf")  # Return infinity for failed computations


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate STYLE model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--domains", nargs="+", required=True, help="List of domains to evaluate on"
    )
    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Initialize components
    dataset_manager = DatasetManager()
    retriever = Retriever(config)
    llm = LLMIntegration(config)  # Pass config to LLM integration
    monitor = Monitor(
        project_name="STYLE_Evaluation", use_wandb=False
    )  # Disable wandb for evaluation

    # Add Config to safe globals for loading
    torch.serialization.add_safe_globals([Config])

    # Load trained model
    try:
        checkpoint = torch.load(args.model_path, map_location=config.DEVICE)
    except Exception as e:
        print(f"Error loading checkpoint with safe globals: {e}")
        print("Trying to load with weights_only=False...")
        checkpoint = torch.load(
            args.model_path, map_location=config.DEVICE, weights_only=False
        )

    disp = DISP(config)
    disp.load_state_dict(checkpoint["model_state"])
    disp.to(config.DEVICE)
    disp.eval()  # Set to evaluation mode

    # Initialize evaluator
    evaluator = Evaluator(disp, retriever, llm, monitor)

    # Load test data and documents
    results = {}
    for domain in args.domains:
        print(f"\nEvaluating on {domain}...")

        # Load documents for the domain
        domain_lower = domain.lower()
        try:
            # Load test data to get documents
            test_data = (
                dataset_manager.load_clariq_dataset("test")
                if domain_lower == "clariq"
                else dataset_manager.load_opendialkg_dataset("test")
            )

            # Extract and load documents into retriever
            all_docs = []
            for item in test_data:
                all_docs.extend(item.get("documents", []))
            retriever.load_documents(domain_lower, all_docs)
            print(f"Loaded {len(all_docs)} documents for {domain}")
        except Exception as e:
            print(f"Error loading documents for {domain}: {e}")
            continue

        # Get test data loader
        test_loader = dataset_manager.get_domain_data(domain_lower, split="test")
        metrics = evaluator.evaluate_domain(domain, test_loader)
        results[domain] = metrics

        # Print domain results
        print(f"\n{domain} Results:")
        print(f"Recall@5: {metrics['recall@5']:.4f}")
        print(f"Success Rate@3: {metrics['sr@3']:.4f}")
        print(f"Success Rate@5: {metrics['sr@5']:.4f}")
        print(f"Average Turns: {metrics['avg_turns']:.4f}")

    # Compare with paper metrics
    evaluator.compare_with_paper(results)

    # Save results
    results_path = "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
