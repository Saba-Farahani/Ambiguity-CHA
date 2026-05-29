#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FoodKG → Neo4j importer  (compatible with Neo4j 2026.04.0)
"""

import argparse, os
from neo4j import GraphDatabase

NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")  # set NEO4J_PASSWORD env var


def run(driver, cypher: str, label: str = "") -> None:
    if label:
        print(f"  → {label}...")
    with driver.session() as s:
        s.run(cypher)


def count(driver, cypher: str, label: str) -> None:
    with driver.session() as s:
        r = s.run(cypher).single()
        print(f"  ✓ {label}: {r[0]:,}")


# ── Indexes ───────────────────────────────────────────────────────
def create_indexes(driver):
    print("\n[1/6] Creating indexes...")
    for label in ["Condition","ConditionFamily","ConditionAlias",
                  "FoodPhrase","Ingredient","FoodProperty","FoodItem","Nutrient"]:
        run(driver,
            f"CREATE INDEX {label.lower()}_id IF NOT EXISTS FOR (n:{label}) ON (n.id)",
            f"Index {label}.id")


# ── Condition nodes ───────────────────────────────────────────────
def load_conditions(driver):
    print("\n[2/6] Loading condition nodes...")

    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///nodes_condition.csv' AS row
        WITH row WHERE row.label = 'Condition'
        MERGE (n:Condition {id: row.id})
        SET n.name = row.name, n.source = row.source
    """, "Condition leaf nodes")

    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///nodes_condition.csv' AS row
        WITH row WHERE row.label = 'ConditionFamily'
        MERGE (n:ConditionFamily {id: row.id})
        SET n.name = row.name, n.source = row.source
    """, "ConditionFamily nodes")

    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///nodes_condition_alias.csv' AS row
        MERGE (n:ConditionAlias {id: row.id})
        SET n.text = row.text
    """, "ConditionAlias nodes")


# ── Food nodes ────────────────────────────────────────────────────
def load_food_nodes(driver):
    print("\n[3/6] Loading food nodes...")

    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///nodes_food_property.csv' AS row
        MERGE (n:FoodProperty {id: row.id})
        SET n.name           = row.name,
            n.description    = row.description,
            n.threshold      = row.threshold,
            n.unit           = row.unit,
            n.source_name    = row.source_name,
            n.source_url     = row.source_url,
            n.evidence_level = row.evidence_level
    """, "FoodProperty nodes")

    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///nodes_ingredient.csv' AS row
        MERGE (n:Ingredient {id: row.id})
        SET n.name = row.name
    """, "Ingredient nodes")

    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///nodes_food_phrase.csv' AS row
        MERGE (n:FoodPhrase {id: row.id})
        SET n.name = row.name
    """, "FoodPhrase nodes")


# ── Core edges ────────────────────────────────────────────────────
def load_core_edges(driver):
    print("\n[4/6] Loading core edges...")

    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///edges_condition_alias_of.csv' AS row
        MATCH (a {id: row.start_id})
        MATCH (b {id: row.end_id})
        MERGE (a)-[:ALIAS_OF {source: row.source}]->(b)
    """, "ALIAS_OF edges")

    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///edges_condition_isa.csv' AS row
        MATCH (a {id: row.start_id})
        MATCH (b {id: row.end_id})
        MERGE (a)-[:IS_A {source: row.source}]->(b)
    """, "IS_A edges (curated)")

    # UMLS IS_A — only if file exists
    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///edges_condition_isa_umls.csv' AS row
        MATCH (a {id: row.start_id})
        MATCH (b {id: row.end_id})
        MERGE (a)-[:IS_A {source: row.source}]->(b)
    """, "IS_A edges (UMLS)")

    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///edges_food_has_ingredient.csv' AS row
        MATCH (fp:FoodPhrase {id: row.start_id})
        MATCH (ing:Ingredient {id: row.end_id})
        MERGE (fp)-[:HAS_INGREDIENT {source: row.source}]->(ing)
    """, "HAS_INGREDIENT edges")

    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///edges_ingredient_has_property.csv' AS row
        MATCH (ing:Ingredient {id: row.start_id})
        MATCH (prop:FoodProperty {id: row.end_id})
        MERGE (ing)-[:HAS_PROPERTY {source: row.source, evidence_level: row.evidence_level}]->(prop)
    """, "HAS_PROPERTY edges")

    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///edges_property_risky_for_condition.csv' AS row
        MATCH (prop:FoodProperty {id: row.start_id})
        MATCH (cond {id: row.end_id})
        MERGE (prop)-[:RISKY_FOR {
            strength: row.strength,
            source_name: row.source_name,
            source_url: row.source_url,
            evidence_level: row.evidence_level
        }]->(cond)
    """, "RISKY_FOR edges")

    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///edges_property_safe_for_condition.csv' AS row
        MATCH (prop:FoodProperty {id: row.start_id})
        MATCH (cond {id: row.end_id})
        MERGE (prop)-[:SAFE_FOR {
            strength: row.strength,
            source_name: row.source_name,
            source_url: row.source_url,
            evidence_level: row.evidence_level
        }]->(cond)
    """, "SAFE_FOR edges")


# ── Shortcut edges ────────────────────────────────────────────────
def load_shortcuts(driver):
    print("\n[5/6] Loading shortcut edges...")
    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///edges_food_risky_for_condition_shortcut.csv' AS row
        CALL {
            WITH row
            MATCH (fp:FoodPhrase {id: row.start_id})
            MATCH (cond {id: row.end_id})
            MERGE (fp)-[:RISKY_FOR {
                via_prop: row.via_prop,
                strength: row.strength,
                source_name: row.source_name,
                evidence_level: row.evidence_level,
                path: row.path
            }]->(cond)
        } IN TRANSACTIONS OF 500 ROWS
    """, "RISKY_FOR shortcut edges")

    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///edges_food_safe_for_condition_shortcut.csv' AS row
        CALL {
            WITH row
            MATCH (fp:FoodPhrase {id: row.start_id})
            MATCH (cond {id: row.end_id})
            MERGE (fp)-[:SAFE_FOR {
                via_prop: row.via_prop,
                strength: row.strength,
                source_name: row.source_name,
                evidence_level: row.evidence_level,
                path: row.path
            }]->(cond)
        } IN TRANSACTIONS OF 500 ROWS
    """, "SAFE_FOR shortcut edges")


# ── USDA (optional) ───────────────────────────────────────────────
def load_usda(driver):
    print("\n[USDA] Loading USDA nodes and edges (may take 10-20 min)...")

    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///nodes_food_item.csv' AS row
        CALL {
            WITH row
            MERGE (n:FoodItem {id: row.id})
            SET n.fdc_id = toInteger(row.fdc_id),
                n.name = row.name,
                n.data_type = row.data_type
        } IN TRANSACTIONS OF 2000 ROWS
    """, "FoodItem nodes (2M)")

    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///nodes_nutrient.csv' AS row
        MERGE (n:Nutrient {id: row.id})
        SET n.nutrient_id = toInteger(row.nutrient_id),
            n.name = row.name, n.unit = row.unit
    """, "Nutrient nodes")

    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///edges_food_item_has_property.csv' AS row
        CALL {
            WITH row
            MATCH (fi:FoodItem {id: row.start_id})
            MATCH (prop:FoodProperty {id: row.end_id})
            MERGE (fi)-[:HAS_PROPERTY {source: row.source}]->(prop)
        } IN TRANSACTIONS OF 2000 ROWS
    """, "FoodItem→Property edges")

    run(driver, """
        LOAD CSV WITH HEADERS FROM 'file:///edges_food_has_nutrient.csv' AS row
        CALL {
            WITH row
            MATCH (fi:FoodItem {id: row.start_id})
            MATCH (nut:Nutrient {id: row.end_id})
            MERGE (fi)-[:HAS_NUTRIENT {amount: toFloat(row.amount)}]->(nut)
        } IN TRANSACTIONS OF 2000 ROWS
    """, "HAS_NUTRIENT edges (27M)")


# ── Verify ────────────────────────────────────────────────────────
def verify(driver):
    print("\n[6/6] Verification...")
    count(driver, "MATCH (n:FoodPhrase) RETURN count(n)",       "FoodPhrase nodes")
    count(driver, "MATCH (n:Ingredient) RETURN count(n)",       "Ingredient nodes")
    count(driver, "MATCH (n:FoodProperty) RETURN count(n)",     "FoodProperty nodes")
    count(driver, "MATCH (n:Condition) RETURN count(n)",        "Condition nodes")
    count(driver, "MATCH (n:ConditionFamily) RETURN count(n)",  "ConditionFamily nodes")
    count(driver, "MATCH ()-[:HAS_INGREDIENT]->() RETURN count(*)", "HAS_INGREDIENT")
    count(driver, "MATCH ()-[:HAS_PROPERTY]->() RETURN count(*)",   "HAS_PROPERTY")
    count(driver, "MATCH ()-[:RISKY_FOR]->() RETURN count(*)",      "RISKY_FOR total")
    count(driver, "MATCH ()-[:SAFE_FOR]->() RETURN count(*)",       "SAFE_FOR")
    count(driver, "MATCH ()-[:IS_A]->() RETURN count(*)",           "IS_A")

    print("\n  Sample — white rice risks:")
    with driver.session() as s:
        rows = s.run("""
            MATCH (fp:FoodPhrase {name:'white rice'})-[:RISKY_FOR]->(c)
            RETURN DISTINCT c.name AS c ORDER BY c
        """)
        for r in rows:
            print(f"    • {r['c']}")


# ── Main ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_usda", action="store_true",
                    help="Skip USDA 2M food items and 27M edges (fast, recommended for testing)")
    ap.add_argument("--verify",    action="store_true",
                    help="Only run verification, skip import")
    args = ap.parse_args()

    print(f"FoodKG → Neo4j  |  URI: {NEO4J_URI}  |  USDA: {'skip' if args.skip_usda else 'load'}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        driver.verify_connectivity()
        print("✓ Connected\n")
    except Exception as e:
        print(f"✗ Cannot connect: {e}"); return

    if not args.verify:
        create_indexes(driver)
        load_conditions(driver)
        load_food_nodes(driver)
        load_core_edges(driver)
        load_shortcuts(driver)
        if not args.skip_usda:
            load_usda(driver)

    verify(driver)
    driver.close()
    print("\n✅ Done.")

if __name__ == "__main__":
    main()
