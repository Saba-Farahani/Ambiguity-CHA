"""
User simulator for generating responses during training.
"""

import os
import random
from typing import List, Dict, Any


class UserSimulator:
    def __init__(self):
        """Initialize the user simulator."""
        self.behaviors = {
            "cooperative": self._generate_cooperative_response,
            "uncertain": self._generate_uncertain_response,
            "expert": self._generate_expert_response,
        }

    def generate_response(
        self,
        query_history: List[str],
        system_responses: List[str],
        current_context: Dict[str, Any],
        domain: str,
        user_behavior: str = "cooperative",
    ) -> str:
        """
        Generate a user response based on the conversation history and context.

        Args:
            query_history: List of previous user queries
            system_responses: List of system responses
            current_context: Current conversation context
            domain: Current domain (travel, restaurant, movie)
            user_behavior: Type of user behavior to simulate

        Returns:
            str: Generated user response
        """
        if user_behavior not in self.behaviors:
            user_behavior = "cooperative"

        return self.behaviors[user_behavior](
            query_history, system_responses, current_context, domain
        )

    def _generate_cooperative_response(
        self,
        query_history: List[str],
        system_responses: List[str],
        current_context: Dict[str, Any],
        domain: str,
    ) -> str:
        """Generate a cooperative user response."""
        if domain == "travel":
            responses = [
                "That sounds great! Can you tell me more about the location?",
                "I'd like to know more about the prices.",
                "What are the best times to visit?",
                "Are there any special events happening?",
                "That's helpful, thank you!",
            ]
        elif domain == "restaurant":
            responses = [
                "What's the price range?",
                "Do they have vegetarian options?",
                "What are their opening hours?",
                "Is there parking available?",
                "Sounds good, I'll check it out!",
            ]
        else:  # movie domain
            responses = [
                "What are the showtimes?",
                "How much are the tickets?",
                "Is it family-friendly?",
                "What's the rating?",
                "Great, I'll book tickets!",
            ]

        return random.choice(responses)

    def _generate_uncertain_response(
        self,
        query_history: List[str],
        system_responses: List[str],
        current_context: Dict[str, Any],
        domain: str,
    ) -> str:
        """Generate an uncertain user response."""
        if domain == "travel":
            responses = [
                "I'm not sure what I'm looking for...",
                "Could you suggest some options?",
                "I'm not familiar with the area...",
                "What would you recommend?",
                "I'm a bit confused...",
            ]
        elif domain == "restaurant":
            responses = [
                "I'm not sure what I want to eat...",
                "What's popular here?",
                "I'm not familiar with this cuisine...",
                "What would you recommend?",
                "I'm not sure about the prices...",
            ]
        else:  # movie domain
            responses = [
                "I'm not sure what to watch...",
                "What's popular right now?",
                "I'm not familiar with these movies...",
                "What would you recommend?",
                "I'm not sure about the timing...",
            ]

        return random.choice(responses)

    def _generate_expert_response(
        self,
        query_history: List[str],
        system_responses: List[str],
        current_context: Dict[str, Any],
        domain: str,
    ) -> str:
        """Generate an expert user response."""
        if domain == "travel":
            responses = [
                "I need specific details about the location and amenities.",
                "What are the exact prices and availability?",
                "I want to know about the local transportation options.",
                "Please provide information about nearby attractions.",
                "I need to know about the booking policies.",
            ]
        elif domain == "restaurant":
            responses = [
                "I need the exact menu and pricing details.",
                "What are the specific dietary options available?",
                "I want to know about the reservation policies.",
                "Please provide information about the chef's specialties.",
                "I need to know about the wine selection.",
            ]
        else:  # movie domain
            responses = [
                "I need the exact showtimes and seat availability.",
                "What are the specific ticket prices and discounts?",
                "I want to know about the theater amenities.",
                "Please provide information about the movie ratings.",
                "I need to know about the booking policies.",
            ]

        return random.choice(responses)
