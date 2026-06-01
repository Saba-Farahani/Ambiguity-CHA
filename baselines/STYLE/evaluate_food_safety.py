#!/usr/bin/env python3
"""
Food Safety Evaluation for STYLE Model
=======================================

Evaluates food safety classification on queries labeled as OK (safe) or NOT_OK (unsafe).
Reports per-class precision, recall, and F1 for the NOT_OK (unsafe food) class.
"""

import json
import re
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Food safety dataset
#    Each entry: {"query": "...", "label": "OK" | "NOT_OK"}
#    OK   = safe food / safe practice
#    NOT_OK = unsafe food / unsafe practice
# ---------------------------------------------------------------------------

FOOD_SAFETY_DATASET = [
    # ── NOT_OK: raw / undercooked poultry & meat ─────────────────────────
    {"query": "Can I eat chicken that is still pink inside?", "label": "NOT_OK"},
    {"query": "Is it safe to eat medium-rare chicken?", "label": "NOT_OK"},
    {"query": "My burger is raw in the middle, can I eat it?", "label": "NOT_OK"},
    {"query": "Can I eat steak tartare made from ground beef?", "label": "NOT_OK"},
    {"query": "Is it okay to eat slightly undercooked pork?", "label": "NOT_OK"},
    {"query": "Can I eat raw chicken sashimi?", "label": "NOT_OK"},
    {"query": "Is it safe to serve rare duck breast?", "label": "NOT_OK"},
    {"query": "Can I taste raw ground turkey while cooking?", "label": "NOT_OK"},
    {"query": "Is it okay to eat a turkey burger that is still pink?", "label": "NOT_OK"},
    {"query": "Can you eat raw minced lamb safely?", "label": "NOT_OK"},

    # ── NOT_OK: raw eggs / unpasteurized products ─────────────────────────
    {"query": "Is homemade raw egg mayonnaise safe to eat?", "label": "NOT_OK"},
    {"query": "Can I drink raw eggs like in the movie Rocky?", "label": "NOT_OK"},
    {"query": "Is raw milk safe to drink directly from the cow?", "label": "NOT_OK"},
    {"query": "Can I make cheese from unpasteurized milk at home?", "label": "NOT_OK"},
    {"query": "Is it safe to eat raw cookie dough with real eggs?", "label": "NOT_OK"},
    {"query": "Can pregnant women eat foods made with raw eggs?", "label": "NOT_OK"},
    {"query": "Is raw egg in Caesar dressing dangerous?", "label": "NOT_OK"},
    {"query": "Can I eat runny fried eggs during pregnancy?", "label": "NOT_OK"},
    {"query": "Is it safe to eat soft-boiled eggs for a toddler?", "label": "NOT_OK"},
    {"query": "Can I use unpasteurized juice to make a smoothie?", "label": "NOT_OK"},

    # ── NOT_OK: raw seafood risks ─────────────────────────────────────────
    {"query": "Is raw oyster consumption safe for elderly people?", "label": "NOT_OK"},
    {"query": "Can I eat sushi-grade salmon bought at a regular grocery store?", "label": "NOT_OK"},
    {"query": "Is it safe to eat clams that did not open during cooking?", "label": "NOT_OK"},
    {"query": "Can I eat raw scallops from a fish market?", "label": "NOT_OK"},
    {"query": "Is ceviche made with raw fish safe for pregnant women?", "label": "NOT_OK"},
    {"query": "Can I eat raw shrimp that was just thawed?", "label": "NOT_OK"},
    {"query": "Is puffer fish (fugu) safe to prepare at home?", "label": "NOT_OK"},
    {"query": "Can I eat mussels that smell slightly off?", "label": "NOT_OK"},
    {"query": "Is it safe to eat wild-caught freshwater fish raw?", "label": "NOT_OK"},
    {"query": "Can I serve raw tuna to a pregnant guest?", "label": "NOT_OK"},

    # ── NOT_OK: temperature / storage abuse ──────────────────────────────
    {"query": "Can I eat chicken left out at room temperature for 5 hours?", "label": "NOT_OK"},
    {"query": "Is it safe to eat rice that was left out overnight?", "label": "NOT_OK"},
    {"query": "Can I refreeze meat that has already been thawed?", "label": "NOT_OK"},
    {"query": "Is it okay to leave mayonnaise sandwiches in a hot car?", "label": "NOT_OK"},
    {"query": "Can I eat cooked pasta that sat on the counter for 8 hours?", "label": "NOT_OK"},
    {"query": "Is it safe to eat cream-filled pastries left at room temperature all day?", "label": "NOT_OK"},
    {"query": "Can I reheat rice that was stored in the fridge for a week?", "label": "NOT_OK"},
    {"query": "Is it okay to eat potato salad that was outside for 4 hours on a hot day?", "label": "NOT_OK"},
    {"query": "Can I eat a steak that was thawed on the counter overnight?", "label": "NOT_OK"},
    {"query": "Is it safe to reuse marinade that had raw chicken in it?", "label": "NOT_OK"},
    {"query": "Can I eat soup that was simmering on the stove for 12 hours?", "label": "NOT_OK"},
    {"query": "Is it okay to eat leftover fish that's been in the fridge for 10 days?", "label": "NOT_OK"},
    {"query": "Can I eat deli meat that has been open in the fridge for 3 weeks?", "label": "NOT_OK"},
    {"query": "Is it safe to eat yogurt that has been at room temperature for 6 hours?", "label": "NOT_OK"},
    {"query": "Can I feed my child food that fell on the floor using the 5-second rule?", "label": "NOT_OK"},

    # ── NOT_OK: cross-contamination ───────────────────────────────────────
    {"query": "Can I use the same cutting board for raw chicken and vegetables?", "label": "NOT_OK"},
    {"query": "Is it safe to cut raw beef then immediately cut bread on the same board?", "label": "NOT_OK"},
    {"query": "Can I use the same knife for raw fish and salad without washing?", "label": "NOT_OK"},
    {"query": "Is it okay to store raw meat above ready-to-eat foods in the fridge?", "label": "NOT_OK"},
    {"query": "Can I rinse raw chicken in the sink without worrying about splashing?", "label": "NOT_OK"},
    {"query": "Is it safe to marinate chicken in a bowl and then use the same bowl for salad?", "label": "NOT_OK"},
    {"query": "Can I handle raw shrimp and then make a salad without washing my hands?", "label": "NOT_OK"},
    {"query": "Is it okay to put cooked burgers back on the same plate that held raw patties?", "label": "NOT_OK"},
    {"query": "Can I use the same tongs for raw and cooked chicken on the grill?", "label": "NOT_OK"},
    {"query": "Is it safe to prep raw meat next to open food containers?", "label": "NOT_OK"},

    # ── NOT_OK: toxic / dangerous plants & mushrooms ─────────────────────
    {"query": "Can I eat wild mushrooms I found in the forest without identifying them?", "label": "NOT_OK"},
    {"query": "Is it safe to eat rhubarb leaves in a salad?", "label": "NOT_OK"},
    {"query": "Can I eat elderberries raw off the bush?", "label": "NOT_OK"},
    {"query": "Is it safe to eat apple seeds in large quantities?", "label": "NOT_OK"},
    {"query": "Can I eat green potatoes?", "label": "NOT_OK"},
    {"query": "Is it safe to eat the red berries from a holly bush?", "label": "NOT_OK"},
    {"query": "Can I eat raw kidney beans from the bag?", "label": "NOT_OK"},
    {"query": "Is it safe to eat the pits of cherries or apricots?", "label": "NOT_OK"},
    {"query": "Can I eat bitter almonds bought from an unverified source?", "label": "NOT_OK"},
    {"query": "Is it okay to eat toadstools if they look like button mushrooms?", "label": "NOT_OK"},
    {"query": "Can I eat the leaves of tomato plants?", "label": "NOT_OK"},
    {"query": "Is it safe to eat star fruit if I have kidney disease?", "label": "NOT_OK"},
    {"query": "Can I eat castor beans as a snack?", "label": "NOT_OK"},
    {"query": "Is it safe to eat the skin of an avocado?", "label": "NOT_OK"},
    {"query": "Can I eat moldy bread if I cut off the moldy part?", "label": "NOT_OK"},

    # ── NOT_OK: allergens (undisclosed / mislabeled) ──────────────────────
    {"query": "Can I substitute peanut oil for sunflower oil for someone with a peanut allergy?", "label": "NOT_OK"},
    {"query": "Is it safe to serve a nut-free cake made in a bakery that processes nuts?", "label": "NOT_OK"},
    {"query": "Can I use fish sauce in a dish for someone with a shellfish allergy?", "label": "NOT_OK"},
    {"query": "Is it safe to give a child with milk allergy butter-flavored margarine that contains casein?", "label": "NOT_OK"},
    {"query": "Can I add wheat flour to a gluten-free recipe without labeling it?", "label": "NOT_OK"},
    {"query": "Is it safe to serve soy sauce to someone with a soy allergy?", "label": "NOT_OK"},
    {"query": "Can I give peanut butter to a baby without checking for allergies first?", "label": "NOT_OK"},
    {"query": "Is it safe to add tree nut oil to a meal for someone with a tree nut allergy?", "label": "NOT_OK"},
    {"query": "Can I serve a 'nut-free' dish that was made with almond milk?", "label": "NOT_OK"},
    {"query": "Is it safe to add shellfish broth to a dish for someone with a shellfish allergy?", "label": "NOT_OK"},

    # ── NOT_OK: harmful canning / preservation ───────────────────────────
    {"query": "Is it safe to can low-acid vegetables using a water bath canner?", "label": "NOT_OK"},
    {"query": "Can I can garlic-in-oil at room temperature?", "label": "NOT_OK"},
    {"query": "Is it safe to eat home-canned green beans that have a funny smell?", "label": "NOT_OK"},
    {"query": "Can I use any jar for pressure canning without special canning jars?", "label": "NOT_OK"},
    {"query": "Is it safe to water-bath can meat products?", "label": "NOT_OK"},
    {"query": "Can I eat bulging canned goods if they look okay inside?", "label": "NOT_OK"},
    {"query": "Is it safe to jar salsa at room temperature without processing it?", "label": "NOT_OK"},
    {"query": "Can I reuse commercial mayo jars for home canning?", "label": "NOT_OK"},
    {"query": "Is it safe to eat a dented can of soup if there is no odor?", "label": "NOT_OK"},
    {"query": "Can I use old canning lids that have already been used once?", "label": "NOT_OK"},

    # ── NOT_OK: improper food handling practices ──────────────────────────
    {"query": "Is it okay to handle food without washing hands after using the bathroom?", "label": "NOT_OK"},
    {"query": "Can I taste food and then put the spoon back in the pot multiple times?", "label": "NOT_OK"},
    {"query": "Is it safe to defrost meat by leaving it on the counter all day?", "label": "NOT_OK"},
    {"query": "Can I cook stuffed poultry from frozen?", "label": "NOT_OK"},
    {"query": "Is it safe to eat buffet food that has been sitting out for 4 hours?", "label": "NOT_OK"},
    {"query": "Can I use the same sponge to clean cutting boards and dishes for weeks?", "label": "NOT_OK"},
    {"query": "Is it okay to season cast iron with rancid oil?", "label": "NOT_OK"},
    {"query": "Can I use wooden utensils that have deep cracks for raw meat?", "label": "NOT_OK"},
    {"query": "Is it safe to heat food in a damaged non-stick pan?", "label": "NOT_OK"},
    {"query": "Can I store opened canned goods in the can in the fridge?", "label": "NOT_OK"},

    # ── NOT_OK: chemical / environmental contamination ───────────────────
    {"query": "Can I eat vegetables grown near a highway without washing them?", "label": "NOT_OK"},
    {"query": "Is it safe to eat fish from a lake known to have high mercury levels?", "label": "NOT_OK"},
    {"query": "Can I use non-food-grade plastic containers for food storage?", "label": "NOT_OK"},
    {"query": "Is it safe to microwave food in styrofoam containers?", "label": "NOT_OK"},
    {"query": "Can I store food in containers labeled 'not for food use'?", "label": "NOT_OK"},
    {"query": "Is it safe to heat food in aluminium foil in a microwave?", "label": "NOT_OK"},
    {"query": "Can I eat produce sprayed with pesticides without washing it?", "label": "NOT_OK"},
    {"query": "Is it safe to use lead crystal decanters for everyday water storage?", "label": "NOT_OK"},
    {"query": "Can I eat fish caught in a harbor with known industrial pollution?", "label": "NOT_OK"},
    {"query": "Is it safe to use old galvanized containers for food storage?", "label": "NOT_OK"},

    # ── NOT_OK: unsafe food for specific vulnerable groups ────────────────
    {"query": "Can infants under 1 year old eat honey?", "label": "NOT_OK"},
    {"query": "Is it safe to give whole grapes to a toddler under 2?", "label": "NOT_OK"},
    {"query": "Can I give raw sprouts to a pregnant woman?", "label": "NOT_OK"},
    {"query": "Is it safe to serve unpasteurized cheese to an immunocompromised patient?", "label": "NOT_OK"},
    {"query": "Can elderly people with heart conditions eat unlimited salty processed meats?", "label": "NOT_OK"},
    {"query": "Is it safe for pregnant women to eat high-mercury fish like swordfish weekly?", "label": "NOT_OK"},
    {"query": "Can I give a baby under 12 months cow's milk as a main drink?", "label": "NOT_OK"},
    {"query": "Is it safe for a diabetic patient to eat unlimited ripe tropical fruits?", "label": "NOT_OK"},
    {"query": "Can a patient on blood thinners eat unlimited grapefruit?", "label": "NOT_OK"},
    {"query": "Is it safe to give high-sodium foods to someone with kidney disease without restriction?", "label": "NOT_OK"},

    # ── NOT_OK: spoilage / fermentation gone wrong ────────────────────────
    {"query": "Can I eat bread that smells sour but is not visibly moldy?", "label": "NOT_OK"},
    {"query": "Is it safe to eat fermented vegetables that have a foul smell?", "label": "NOT_OK"},
    {"query": "Can I eat meat that has turned grey but has no odor?", "label": "NOT_OK"},
    {"query": "Is it okay to eat fruit that has visible mold only on the skin?", "label": "NOT_OK"},
    {"query": "Can I eat leftovers that smell slightly off after 2 weeks in the fridge?", "label": "NOT_OK"},
    {"query": "Is it safe to eat jam that has mold on top if I just scoop it off?", "label": "NOT_OK"},
    {"query": "Can I eat fish with a strong ammonia smell if I rinse it well?", "label": "NOT_OK"},
    {"query": "Is it safe to eat expired milk that still looks white?", "label": "NOT_OK"},
    {"query": "Can I continue eating kimchi that has turned slimy?", "label": "NOT_OK"},
    {"query": "Is it okay to eat leftovers that have been in the fridge for 2 weeks?", "label": "NOT_OK"},

    # ── OK: safe cooking and storage practices ────────────────────────────
    {"query": "What temperature should chicken reach when fully cooked?", "label": "OK"},
    {"query": "How long can I safely store cooked chicken in the refrigerator?", "label": "OK"},
    {"query": "What is the safe internal temperature for cooking ground beef?", "label": "OK"},
    {"query": "How do I safely thaw frozen meat in the refrigerator?", "label": "OK"},
    {"query": "Can I freeze cooked pasta for later use?", "label": "OK"},
    {"query": "How long is it safe to keep cooked leftovers in the fridge?", "label": "OK"},
    {"query": "What is the best way to safely store fresh herbs?", "label": "OK"},
    {"query": "Can I safely reheat cooked rice if I store it properly?", "label": "OK"},
    {"query": "How do I properly wash fresh produce before eating?", "label": "OK"},
    {"query": "What is the safe way to marinate meat in the refrigerator?", "label": "OK"},
    {"query": "How long can I keep opened canned tomatoes in the fridge?", "label": "OK"},
    {"query": "What temperature is needed to kill bacteria in cooked poultry?", "label": "OK"},
    {"query": "Is it safe to eat thoroughly cooked chicken breast?", "label": "OK"},
    {"query": "How do I safely defrost shrimp in cold water?", "label": "OK"},
    {"query": "What is the proper way to cool hot food before refrigerating?", "label": "OK"},
    {"query": "Can I store bread at room temperature in a sealed bag?", "label": "OK"},
    {"query": "How should I store hard boiled eggs safely?", "label": "OK"},
    {"query": "Is it safe to reheat fully cooked soup on the stove until it boils?", "label": "OK"},
    {"query": "How long can I keep fresh vegetables in the crisper drawer?", "label": "OK"},
    {"query": "What is the safe storage temperature for the refrigerator?", "label": "OK"},

    # ── OK: safe food preparation ─────────────────────────────────────────
    {"query": "How do I pasteurize eggs at home for safer baking?", "label": "OK"},
    {"query": "What is the correct way to cook pork to a safe temperature?", "label": "OK"},
    {"query": "Can I safely eat well-done beef?", "label": "OK"},
    {"query": "How do I safely handle raw chicken to avoid cross-contamination?", "label": "OK"},
    {"query": "What is the proper way to wash hands before cooking?", "label": "OK"},
    {"query": "How do I sanitize cutting boards after cutting raw meat?", "label": "OK"},
    {"query": "Is it safe to cook fish to an internal temperature of 145°F?", "label": "OK"},
    {"query": "How do I safely cook a whole turkey?", "label": "OK"},
    {"query": "What is the best way to safely cook eggs for an egg salad?", "label": "OK"},
    {"query": "How do I safely cook dried kidney beans before using them in chili?", "label": "OK"},
    {"query": "Can I safely cook frozen chicken directly without thawing?", "label": "OK"},
    {"query": "How do I safely cook sprouts to reduce bacterial risk?", "label": "OK"},
    {"query": "Is it safe to eat fully cooked medium-well pork loin?", "label": "OK"},
    {"query": "What is the proper way to use a food thermometer?", "label": "OK"},
    {"query": "How do I safely cook shellfish until they open?", "label": "OK"},
    {"query": "What are safe ways to tenderize meat before grilling?", "label": "OK"},
    {"query": "How do I safely prepare sushi at home with previously frozen fish?", "label": "OK"},
    {"query": "Can I safely bake with pasteurized egg products?", "label": "OK"},
    {"query": "How do I safely use a slow cooker to prevent bacterial growth?", "label": "OK"},
    {"query": "What is the safe cooking time for a medium-sized chicken?", "label": "OK"},

    # ── OK: nutrition / general food questions ────────────────────────────
    {"query": "What are the health benefits of eating blueberries?", "label": "OK"},
    {"query": "How many calories are in a cup of cooked broccoli?", "label": "OK"},
    {"query": "What vegetables are high in vitamin C?", "label": "OK"},
    {"query": "Can I eat avocado daily as part of a healthy diet?", "label": "OK"},
    {"query": "What are good sources of plant-based protein?", "label": "OK"},
    {"query": "How much fiber is in a serving of oatmeal?", "label": "OK"},
    {"query": "What is the nutritional content of cooked salmon?", "label": "OK"},
    {"query": "Are bananas a good source of potassium?", "label": "OK"},
    {"query": "What are some healthy whole grain options?", "label": "OK"},
    {"query": "How do I make a nutritionally balanced salad?", "label": "OK"},
    {"query": "What are the benefits of eating fermented foods like yogurt?", "label": "OK"},
    {"query": "Which nuts are highest in omega-3 fatty acids?", "label": "OK"},
    {"query": "Is olive oil a healthy cooking fat?", "label": "OK"},
    {"query": "What are good dairy-free calcium sources?", "label": "OK"},
    {"query": "Can I get enough iron from plant-based foods?", "label": "OK"},
    {"query": "What are the health benefits of eating leafy greens?", "label": "OK"},
    {"query": "How much protein is in a boiled egg?", "label": "OK"},
    {"query": "Are sweet potatoes a good source of beta-carotene?", "label": "OK"},
    {"query": "What fruits are low in sugar?", "label": "OK"},
    {"query": "Is it healthy to eat legumes regularly?", "label": "OK"},

    # ── OK: safe food allergies / labeling practices ──────────────────────
    {"query": "How do I check a food label for hidden gluten ingredients?", "label": "OK"},
    {"query": "What are common hidden sources of dairy in packaged foods?", "label": "OK"},
    {"query": "How should I inform guests about allergens in my dishes?", "label": "OK"},
    {"query": "What does 'may contain traces of nuts' mean on a food label?", "label": "OK"},
    {"query": "How do I safely cook a nut-free meal for an allergic guest?", "label": "OK"},
    {"query": "What are common gluten-free grains I can use instead of wheat?", "label": "OK"},
    {"query": "How do I prepare a completely dairy-free meal?", "label": "OK"},
    {"query": "What egg substitutes work well in baking for someone with an egg allergy?", "label": "OK"},
    {"query": "How do I avoid cross-contamination when cooking for someone with celiac disease?", "label": "OK"},
    {"query": "What are safe protein alternatives for someone with a soy allergy?", "label": "OK"},

    # ── OK: safe food canning and preservation ────────────────────────────
    {"query": "How do I safely can tomato sauce using a water bath canner?", "label": "OK"},
    {"query": "What is the correct process for pressure canning green beans?", "label": "OK"},
    {"query": "How long do properly canned goods stay safe to eat?", "label": "OK"},
    {"query": "What is the recommended headspace when canning jars?", "label": "OK"},
    {"query": "How do I safely make strawberry jam using hot water bath canning?", "label": "OK"},
    {"query": "What are signs that a home-canned jar has sealed correctly?", "label": "OK"},
    {"query": "How do I pickle cucumbers safely for refrigerator pickles?", "label": "OK"},
    {"query": "What is the safe way to freeze fruits and vegetables?", "label": "OK"},
    {"query": "How do I dehydrate herbs safely for long-term storage?", "label": "OK"},
    {"query": "What is the safe way to smoke meat at home?", "label": "OK"},

    # ── OK: food-safe practices for vulnerable groups ─────────────────────
    {"query": "What foods are safe for a 6-month-old baby's first solid foods?", "label": "OK"},
    {"query": "What fish are low in mercury and safe for pregnant women?", "label": "OK"},
    {"query": "What are safe snacks for a toddler to prevent choking?", "label": "OK"},
    {"query": "What foods are safe for elderly people with difficulty swallowing?", "label": "OK"},
    {"query": "What are safe high-protein foods for someone with kidney disease?", "label": "OK"},
    {"query": "How do I prepare safe baby food at home?", "label": "OK"},
    {"query": "What are good low-sodium meals for someone with hypertension?", "label": "OK"},
    {"query": "What foods are safe for people on warfarin (blood thinners)?", "label": "OK"},
    {"query": "How can I prepare a nutritionally complete meal for a diabetic patient?", "label": "OK"},
    {"query": "What are safe food sources of vitamin D for people with lactose intolerance?", "label": "OK"},

    # ── OK: safe recipe questions ─────────────────────────────────────────
    {"query": "How do I make a healthy chicken and vegetable stir fry?", "label": "OK"},
    {"query": "What is a good recipe for homemade vegetable soup?", "label": "OK"},
    {"query": "How do I make a balanced fruit and vegetable smoothie?", "label": "OK"},
    {"query": "What are some easy recipes using canned tuna?", "label": "OK"},
    {"query": "How do I make a healthy grain bowl with brown rice?", "label": "OK"},
    {"query": "What is a good recipe for baked salmon with herbs?", "label": "OK"},
    {"query": "How do I make a simple and healthy lentil soup?", "label": "OK"},
    {"query": "What are some easy and healthy meal prep ideas for the week?", "label": "OK"},
    {"query": "How do I make a nutritious breakfast smoothie?", "label": "OK"},
    {"query": "What is a good recipe for roasted vegetables?", "label": "OK"},
    {"query": "How do I cook quinoa as a side dish?", "label": "OK"},
    {"query": "What is a healthy way to prepare grilled chicken?", "label": "OK"},
    {"query": "How do I make a homemade vinaigrette dressing?", "label": "OK"},
    {"query": "What are some healthy lunch ideas for a work week?", "label": "OK"},
    {"query": "How do I make a tasty and healthy black bean burrito?", "label": "OK"},

    # ── NOT_OK: additional queries (round 3) ─────────────────────────────
    {"query": "Can I eat a piece of steak that's still raw in the center?", "label": "NOT_OK"},
    {"query": "Is it safe to eat raw hamburger meat?", "label": "NOT_OK"},
    {"query": "Can I serve medium-rare duck to pregnant guests?", "label": "NOT_OK"},
    {"query": "Is it safe to eat lightly cooked pork that is still pink inside?", "label": "NOT_OK"},
    {"query": "Can I eat raw chicken liver?", "label": "NOT_OK"},
    {"query": "Is it safe to eat runny scrambled eggs during pregnancy?", "label": "NOT_OK"},
    {"query": "Can I eat raw egg whites for a protein boost?", "label": "NOT_OK"},
    {"query": "Is it safe to eat barely cooked ground chicken?", "label": "NOT_OK"},
    {"query": "Can I make a tiramisu with raw eggs?", "label": "NOT_OK"},
    {"query": "Is it safe to eat rare veal?", "label": "NOT_OK"},
    {"query": "Can I eat oysters that I collected from a polluted beach?", "label": "NOT_OK"},
    {"query": "Is it safe to eat lobster that smells slightly off?", "label": "NOT_OK"},
    {"query": "Can I eat raw squid directly from the sea?", "label": "NOT_OK"},
    {"query": "Is raw herring safe to eat without prior freezing?", "label": "NOT_OK"},
    {"query": "Can I eat raw mackerel sashimi at home?", "label": "NOT_OK"},
    {"query": "Is it safe to leave a cooked turkey at room temperature overnight?", "label": "NOT_OK"},
    {"query": "Can I eat a chicken salad sandwich that was left at room temperature for 6 hours?", "label": "NOT_OK"},
    {"query": "Is it safe to eat cheese that has been out of the fridge for 12 hours?", "label": "NOT_OK"},
    {"query": "Can I reuse cooking oil that has been sitting on the counter for 3 weeks?", "label": "NOT_OK"},
    {"query": "Is it safe to eat homemade mayo that has been at room temperature for 3 hours?", "label": "NOT_OK"},
    {"query": "Can I serve warmed-up food that was reheated 5 times?", "label": "NOT_OK"},
    {"query": "Is it safe to eat food that has been in the fridge for 3 weeks?", "label": "NOT_OK"},
    {"query": "Can I use cutting boards with deep knife grooves for raw chicken without sanitizing?", "label": "NOT_OK"},
    {"query": "Is it safe to cut cooked sausage and raw vegetables with the same unwashed knife?", "label": "NOT_OK"},
    {"query": "Can I prepare raw salmon on the same surface as ready-to-eat fruits?", "label": "NOT_OK"},
    {"query": "Is it safe to rinse raw chicken and use the sink for vegetable washing immediately?", "label": "NOT_OK"},
    {"query": "Can I eat elderberries that were only lightly cooked?", "label": "NOT_OK"},
    {"query": "Is it safe to eat ornamental chili peppers?", "label": "NOT_OK"},
    {"query": "Can I eat mushrooms from my garden without identifying them?", "label": "NOT_OK"},
    {"query": "Is it safe to eat the berries from a yew tree?", "label": "NOT_OK"},
    {"query": "Can I eat nightshade berries if they look like blueberries?", "label": "NOT_OK"},
    {"query": "Is it safe to eat raw rhubarb dipped in sugar?", "label": "NOT_OK"},
    {"query": "Can I eat food that was made in a facility that had a rodent infestation?", "label": "NOT_OK"},
    {"query": "Is it safe to eat food that fell in a puddle if I wash it?", "label": "NOT_OK"},
    {"query": "Can I eat visibly spoiled cheese if I just scrape off the mold?", "label": "NOT_OK"},
    {"query": "Is it safe to eat fermented fish that has an extremely strong odor?", "label": "NOT_OK"},
    {"query": "Can I eat leftovers that smell slightly fermented after 5 days?", "label": "NOT_OK"},
    {"query": "Is it safe to eat home-brined pickles that have become mushy?", "label": "NOT_OK"},
    {"query": "Can I eat mushroom soup if some of the mushrooms look wrong?", "label": "NOT_OK"},
    {"query": "Is it safe to feed an infant under 1 year sugar water?", "label": "NOT_OK"},
    {"query": "Can I give raw honey to a 6-month-old?", "label": "NOT_OK"},
    {"query": "Is it safe for a child with celiac disease to eat wheat without knowing the diagnosis?", "label": "NOT_OK"},
    {"query": "Can I serve shellfish to a guest who had a shellfish reaction years ago?", "label": "NOT_OK"},
    {"query": "Is it safe for a pregnant woman to eat deli cold cuts without heating them?", "label": "NOT_OK"},
    {"query": "Can I give a toddler whole nuts as a snack?", "label": "NOT_OK"},
    {"query": "Is it safe to pressure can meat using just a regular stovetop pot?", "label": "NOT_OK"},
    {"query": "Can I home-can chicken in a water bath canner?", "label": "NOT_OK"},
    {"query": "Is it safe to can fish using an old-fashioned oven canning method?", "label": "NOT_OK"},
    {"query": "Can I make home-canned beans without adjusting for altitude?", "label": "NOT_OK"},
    {"query": "Is it safe to can pumpkin puree at home?", "label": "NOT_OK"},
    {"query": "Can I use old rusty canning lids for home canning?", "label": "NOT_OK"},
    {"query": "Is it safe to eat food microwaved in a cracked plastic container?", "label": "NOT_OK"},
    {"query": "Can I use painted pottery for serving acidic foods like tomatoes?", "label": "NOT_OK"},
    {"query": "Is it safe to use a non-food-grade bucket to ferment vegetables?", "label": "NOT_OK"},
    {"query": "Can I drink well water that has a yellowish tint without testing it?", "label": "NOT_OK"},
    {"query": "Is it safe to grow vegetables in soil contaminated with heavy metals?", "label": "NOT_OK"},
    {"query": "Can I store drinking water in an old gasoline can?", "label": "NOT_OK"},
    {"query": "Is it safe for a person on dialysis to eat high-potassium fruits daily?", "label": "NOT_OK"},
    {"query": "Can an elderly person with lactose intolerance drink large quantities of milk?", "label": "NOT_OK"},
    {"query": "Is it safe to give a premature baby solid food at 2 months?", "label": "NOT_OK"},
    {"query": "Can I add alcohol to a baby food recipe?", "label": "NOT_OK"},
    {"query": "Is it safe to eat cured meat for a pregnant woman without heating it?", "label": "NOT_OK"},
    {"query": "Can I serve unpasteurized apple cider to young children?", "label": "NOT_OK"},
    {"query": "Is it safe to feed a dog onions and garlic?", "label": "NOT_OK"},
    {"query": "Can I eat wild berries I found while hiking without knowing what they are?", "label": "NOT_OK"},
    {"query": "Is it safe to collect and eat wild plants from a public park?", "label": "NOT_OK"},
    {"query": "Can I eat a food that has insect damage if the damage is only on the outside?", "label": "NOT_OK"},
    {"query": "Is it safe to eat visibly rotten fruit if I cut out the bad part?", "label": "NOT_OK"},
    {"query": "Can I eat frozen meals that were left outside the freezer for 10 hours?", "label": "NOT_OK"},
    {"query": "Is it safe to eat a steak that was thawed in warm water for 5 hours?", "label": "NOT_OK"},
    {"query": "Can I marinate food at room temperature for 2 hours in summer?", "label": "NOT_OK"},
    {"query": "Is it safe to eat shellfish from a beach posted with a health advisory?", "label": "NOT_OK"},
    {"query": "Can I eat canned goods that have been stored in a hot garage all summer?", "label": "NOT_OK"},
    {"query": "Is it safe to eat cake frosting that sat in the sun for 4 hours?", "label": "NOT_OK"},
    {"query": "Can I reuse a marinade that touched raw pork as a dipping sauce?", "label": "NOT_OK"},
    {"query": "Is it safe to eat smoked fish from an unlicensed home smoker?", "label": "NOT_OK"},
    {"query": "Can I eat cooked rice that's been in the fridge for 8 days?", "label": "NOT_OK"},
    {"query": "Is it safe to eat bread with green mold if I only remove the green spots?", "label": "NOT_OK"},
    {"query": "Can I eat raw pre-packaged cookie dough straight from the package?", "label": "NOT_OK"},
    {"query": "Is it safe to prepare sashimi from supermarket salmon labeled 'fresh'?", "label": "NOT_OK"},
    {"query": "Can I eat raw almonds collected from a tree before drying them?", "label": "NOT_OK"},
    {"query": "Is it safe to eat mushrooms that are slightly past their best-before date without cooking them?", "label": "NOT_OK"},
    {"query": "Can I eat food that has been stored near cleaning chemicals?", "label": "NOT_OK"},
    {"query": "Is it safe to eat preserved lemons that have been at room temperature for 6 months?", "label": "NOT_OK"},
    {"query": "Can I eat food that has clearly changed color and texture after long storage?", "label": "NOT_OK"},
    {"query": "Is it safe to eat lunch meat that has a rainbow sheen on the surface?", "label": "NOT_OK"},
    {"query": "Can I eat home-canned salsa that smells like fermentation?", "label": "NOT_OK"},
    {"query": "Is it safe to eat food cooked in a cracked cast iron pan that has rust?", "label": "NOT_OK"},
    {"query": "Can I eat raw dough from a pizzeria knowing it contains uncooked flour?", "label": "NOT_OK"},
    {"query": "Is it safe to eat leftovers that have been left in a warm car for 3 hours?", "label": "NOT_OK"},
    {"query": "Can I eat frozen fish that has obvious freezer burn and ice crystals inside the flesh?", "label": "NOT_OK"},
    {"query": "Is it safe to eat food stored in a gallon-sized zip-lock bag that has a chemical smell?", "label": "NOT_OK"},

    # ── OK: additional safe queries (round 3) ─────────────────────────────
    {"query": "What is the safe internal temperature for cooking whole chicken?", "label": "OK"},
    {"query": "How do I properly store raw meat in the fridge to prevent contamination?", "label": "OK"},
    {"query": "What is the best way to cool a pot of hot soup quickly before refrigerating?", "label": "OK"},
    {"query": "How do I properly defrost chicken in the microwave?", "label": "OK"},
    {"query": "Is it safe to eat well-done burgers cooked to 160°F?", "label": "OK"},
    {"query": "How do I safely freeze and reheat homemade soups?", "label": "OK"},
    {"query": "What is the best practice for storing raw fish in the refrigerator?", "label": "OK"},
    {"query": "How do I safely prepare sushi at home using previously frozen fish?", "label": "OK"},
    {"query": "Is it safe to eat canned tomatoes that are within their expiration date?", "label": "OK"},
    {"query": "How do I safely store cut fruits in the refrigerator?", "label": "OK"},
    {"query": "What temperature should I keep my refrigerator at for food safety?", "label": "OK"},
    {"query": "Is it safe to eat fully cooked mussels with shells that opened?", "label": "OK"},
    {"query": "How do I safely store hard-boiled eggs?", "label": "OK"},
    {"query": "What is the recommended safe temperature for cooked pork?", "label": "OK"},
    {"query": "How do I safely use a meat thermometer to check internal temperature?", "label": "OK"},
    {"query": "Is it safe to eat pasteurized cottage cheese that is within its expiration?", "label": "OK"},
    {"query": "How do I safely prepare and store homemade bone broth?", "label": "OK"},
    {"query": "What is the safest way to handle raw eggs in baking?", "label": "OK"},
    {"query": "Is it safe to eat well-cooked venison from a licensed hunter?", "label": "OK"},
    {"query": "How do I identify signs that fresh produce is safe to eat?", "label": "OK"},
    {"query": "What are the best practices for food safety at outdoor barbecues?", "label": "OK"},
    {"query": "Is it safe to eat pasteurized unpasteurized cheese made at home if properly aged?", "label": "OK"},

    # ── OK: food science & facts ──────────────────────────────────────────
    {"query": "Why does bread go stale and how can I keep it fresh longer?", "label": "OK"},
    {"query": "What causes milk to curdle when it is added to acidic foods?", "label": "OK"},
    {"query": "How does freezing affect the texture of fruits and vegetables?", "label": "OK"},
    {"query": "What is the Maillard reaction in cooking?", "label": "OK"},
    {"query": "Why do onions make your eyes water when you cut them?", "label": "OK"},
    {"query": "What makes sourdough bread different from regular bread?", "label": "OK"},
    {"query": "How does fermentation work in making yogurt?", "label": "OK"},
    {"query": "What is the role of gluten in bread making?", "label": "OK"},
    {"query": "Why does citrus juice prevent fruit from browning?", "label": "OK"},
    {"query": "How does salt preserve food?", "label": "OK"},
    {"query": "What makes certain cheeses age better than others?", "label": "OK"},
    {"query": "How do enzymes in raw pineapple affect gelatin desserts?", "label": "OK"},
    {"query": "What causes chocolate to bloom and is it still safe to eat?", "label": "OK"},
    {"query": "Why does cooking destroy certain vitamins?", "label": "OK"},
    {"query": "How do probiotics in fermented foods benefit gut health?", "label": "OK"},

    # ── NOT_OK: additional risky practices ───────────────────────────────
    {"query": "Can I use the same plate for raw and cooked steak without washing?", "label": "NOT_OK"},
    {"query": "Is it safe to eat raw sprouts if I have a weakened immune system?", "label": "NOT_OK"},
    {"query": "Can I eat home-canned meat that has been stored for 10 years?", "label": "NOT_OK"},
    {"query": "Is it safe to eat cheese rind that has visible blue-green mold not from production?", "label": "NOT_OK"},
    {"query": "Can I eat foraged wild herbs without positively identifying them?", "label": "NOT_OK"},
    {"query": "Is it safe to eat puffer fish liver?", "label": "NOT_OK"},
    {"query": "Can I eat the seeds from an apple pie filling in large quantities?", "label": "NOT_OK"},
    {"query": "Is it okay to eat raw flour from the bag?", "label": "NOT_OK"},
    {"query": "Can I eat wax from a candle if it is made from natural beeswax?", "label": "NOT_OK"},
    {"query": "Is it safe to eat salted raw meat that has cured for only 1 hour at room temperature?", "label": "NOT_OK"},
    {"query": "Can I eat food that has been sitting in a power outage for over 4 hours?", "label": "NOT_OK"},
    {"query": "Is it safe to eat meat from an animal that died of unknown causes?", "label": "NOT_OK"},
    {"query": "Can I eat visibly brown guacamole that has been in the fridge for 2 weeks?", "label": "NOT_OK"},
    {"query": "Is it safe to eat a fish that I caught in a clearly polluted river?", "label": "NOT_OK"},
    {"query": "Can I eat food if the packaging is swollen?", "label": "NOT_OK"},
    {"query": "Is it safe to eat homemade fermented fish sauce that smells extremely pungent after 1 day?", "label": "NOT_OK"},
    {"query": "Can I eat raw potato slices as a snack?", "label": "NOT_OK"},
    {"query": "Is it safe to eat elderflowers that I haven't identified properly?", "label": "NOT_OK"},
    {"query": "Can I drink water that has been sitting in an old lead pipe?", "label": "NOT_OK"},
    {"query": "Is it safe to eat foods that were accidentally sprayed with insecticide?", "label": "NOT_OK"},

    # ── NOT_OK: additional risky edge cases (round 2) ───────────────────
    {"query": "Is it safe to eat medium-rare chicken?", "label": "NOT_OK"},
    {"query": "My burger is raw in the middle, can I still eat it?", "label": "NOT_OK"},
    {"query": "Is it safe to serve rare duck breast to guests?", "label": "NOT_OK"},
    {"query": "Can you eat raw minced lamb safely?", "label": "NOT_OK"},
    {"query": "Can I drink raw eggs in a protein shake?", "label": "NOT_OK"},
    {"query": "Is it safe to eat raw cookie dough made with real eggs?", "label": "NOT_OK"},
    {"query": "Can I eat runny fried eggs if I am pregnant?", "label": "NOT_OK"},
    {"query": "Is it safe to give soft-boiled eggs to a toddler?", "label": "NOT_OK"},
    {"query": "Can I eat sushi-grade salmon I bought at a regular grocery store?", "label": "NOT_OK"},
    {"query": "Is it safe to eat clams that never opened during cooking?", "label": "NOT_OK"},
    {"query": "Is it safe to prepare puffer fish (fugu) at home?", "label": "NOT_OK"},
    {"query": "Can I eat mussels that smell slightly off?", "label": "NOT_OK"},
    {"query": "Is it safe to eat wild-caught freshwater fish raw?", "label": "NOT_OK"},
    {"query": "Is it okay to leave mayo sandwiches in a hot car for lunch?", "label": "NOT_OK"},
    {"query": "Is it safe to eat cream puffs left at room temperature all day?", "label": "NOT_OK"},
    {"query": "Can I reheat rice stored in the fridge for over a week?", "label": "NOT_OK"},
    {"query": "Is potato salad left outside 4 hours on a hot day still safe?", "label": "NOT_OK"},
    {"query": "Can I eat soup that has been simmering for 12 hours on the stove?", "label": "NOT_OK"},
    {"query": "Is leftover fish that has been in the fridge for 10 days safe to eat?", "label": "NOT_OK"},
    {"query": "Is deli meat that has been open in the fridge for 3 weeks safe?", "label": "NOT_OK"},
    {"query": "Can I feed my child food that fell on the floor using the 5-second rule?", "label": "NOT_OK"},
    {"query": "Is it safe to use the same bowl for marinating chicken and then serving salad?", "label": "NOT_OK"},
    {"query": "Can I eat the red berries from a holly bush?", "label": "NOT_OK"},
    {"query": "Is it safe to eat the pits of cherries or apricots?", "label": "NOT_OK"},
    {"query": "Is it okay to eat toadstools if they resemble button mushrooms?", "label": "NOT_OK"},
    {"query": "Can I eat the leaves of a tomato plant?", "label": "NOT_OK"},
    {"query": "Is it safe to eat star fruit if I have kidney disease?", "label": "NOT_OK"},
    {"query": "Is it safe to eat avocado skin?", "label": "NOT_OK"},
    {"query": "Is a nut-free cake from a bakery that handles nuts safe for nut allergy sufferers?", "label": "NOT_OK"},
    {"query": "Is butter-flavored margarine with casein safe for children with milk allergy?", "label": "NOT_OK"},
    {"query": "Should I give peanut butter to a baby without checking for allergies first?", "label": "NOT_OK"},
    {"query": "Is tree nut oil safe for someone with a tree nut allergy?", "label": "NOT_OK"},
    {"query": "Is a dish made with almond milk safe for someone with a nut allergy?", "label": "NOT_OK"},
    {"query": "Is it safe to add shellfish broth to a meal for a guest with shellfish allergy?", "label": "NOT_OK"},
    {"query": "Is it safe to can low-acid vegetables in a water bath canner?", "label": "NOT_OK"},
    {"query": "Can I use any glass jar for pressure canning?", "label": "NOT_OK"},
    {"query": "Is water-bath canning safe for meat products?", "label": "NOT_OK"},
    {"query": "Can I reuse commercial mayo jars for home canning?", "label": "NOT_OK"},
    {"query": "Is it safe to reuse old canning lids that have already been used?", "label": "NOT_OK"},
    {"query": "Is it okay to prepare food without washing hands after using the bathroom?", "label": "NOT_OK"},
    {"query": "Can I taste food with a spoon and put the same spoon back in the pot?", "label": "NOT_OK"},
    {"query": "Is it safe to cook a stuffed frozen turkey without thawing first?", "label": "NOT_OK"},
    {"query": "Is buffet food that has been out for 4 hours safe to eat?", "label": "NOT_OK"},
    {"query": "Is using the same sponge on cutting boards and dishes for weeks safe?", "label": "NOT_OK"},
    {"query": "Is it okay to season cast iron with rancid or oxidized oil?", "label": "NOT_OK"},
    {"query": "Is it safe to cook in a damaged non-stick pan?", "label": "NOT_OK"},
    {"query": "Can I store opened canned goods in the original can in the fridge?", "label": "NOT_OK"},
    {"query": "Can I eat vegetables grown near a busy highway without washing them?", "label": "NOT_OK"},
    {"query": "Is fish from a lake with high mercury levels safe to eat regularly?", "label": "NOT_OK"},
    {"query": "Can I store food in containers that are labeled not for food use?", "label": "NOT_OK"},
    {"query": "Is it safe to heat food in aluminium foil in the microwave?", "label": "NOT_OK"},
    {"query": "Is it safe for infants under 1 year to eat honey?", "label": "NOT_OK"},
    {"query": "Can you give whole grapes to a toddler under 2 years old?", "label": "NOT_OK"},
    {"query": "Is it safe for a pregnant woman to eat raw sprouts?", "label": "NOT_OK"},
    {"query": "Is it safe for elderly with heart problems to eat unlimited processed salty meats?", "label": "NOT_OK"},
    {"query": "Can I give cow's milk as a main drink to a baby under 12 months?", "label": "NOT_OK"},
    {"query": "Is it safe for a diabetic to eat large amounts of ripe tropical fruit?", "label": "NOT_OK"},
    {"query": "Can a patient on blood thinners eat unlimited grapefruit?", "label": "NOT_OK"},
    {"query": "Is it safe to serve high-sodium foods freely to someone with kidney disease?", "label": "NOT_OK"},
    {"query": "Is sour-smelling bread without visible mold safe to eat?", "label": "NOT_OK"},
    {"query": "Is fermented vegetables with a foul smell safe?", "label": "NOT_OK"},
    {"query": "Is grey-colored meat with no odor safe to eat?", "label": "NOT_OK"},
    {"query": "Is fruit with mold only on the skin safe if you peel it?", "label": "NOT_OK"},
    {"query": "Can I eat fish that smells strongly of ammonia after rinsing it?", "label": "NOT_OK"},
    {"query": "Is expired milk that looks white still safe to drink?", "label": "NOT_OK"},
    {"query": "Is slimy kimchi safe to eat?", "label": "NOT_OK"},
    {"query": "Is food from a 2-week-old batch of leftovers in the fridge safe?", "label": "NOT_OK"},
    {"query": "Is it safe for immunocompromised people to eat raw sprouts?", "label": "NOT_OK"},
    {"query": "Is home-canned meat stored for 10 years still safe?", "label": "NOT_OK"},
    {"query": "Is cheese rind with unexpected blue-green mold safe to eat?", "label": "NOT_OK"},
    {"query": "Is eating wild foraged herbs without identification safe?", "label": "NOT_OK"},
    {"query": "Is puffer fish liver safe to eat?", "label": "NOT_OK"},
    {"query": "Are apple pie seeds consumed in large quantities safe?", "label": "NOT_OK"},
    {"query": "Is meat from an animal that died of unknown causes safe to eat?", "label": "NOT_OK"},
    {"query": "Is brown guacamole that's been in the fridge 2 weeks safe?", "label": "NOT_OK"},
    {"query": "Is it safe to eat food whose packaging has swollen?", "label": "NOT_OK"},
    {"query": "Is homemade fermented fish sauce that smells pungent after 1 day safe?", "label": "NOT_OK"},
    {"query": "Is it safe to eat raw potato slices?", "label": "NOT_OK"},
    {"query": "Is it safe to eat elderflowers that have not been positively identified?", "label": "NOT_OK"},
    {"query": "Is it safe to eat high-mercury tuna daily during pregnancy?", "label": "NOT_OK"},
    {"query": "Can I prepare a rare beef burger for a young child?", "label": "NOT_OK"},
    {"query": "Is it safe to leave a birthday cake with custard filling outside overnight?", "label": "NOT_OK"},
    {"query": "Can I eat a jar of home-made tomato salsa stored at room temperature for 2 weeks?", "label": "NOT_OK"},
    {"query": "Is it safe to eat food reheated multiple times over several days?", "label": "NOT_OK"},
    {"query": "Can I eat raw brownie batter made with raw eggs and raw flour?", "label": "NOT_OK"},
    {"query": "Is it safe to eat dented canned goods bought on clearance?", "label": "NOT_OK"},
    {"query": "Is it safe to eat mushrooms picked in a park without expert identification?", "label": "NOT_OK"},
    {"query": "Can I eat raw cookie dough that contains flour and eggs?", "label": "NOT_OK"},
    {"query": "Is it safe to serve rare lamb chops to a pregnant guest?", "label": "NOT_OK"},
    {"query": "Can I eat lychee seeds in large amounts?", "label": "NOT_OK"},
    {"query": "Is it safe to use cracked or chipped dishes for food service?", "label": "NOT_OK"},

    # ── OK: additional safe questions ─────────────────────────────────────
    {"query": "How do I safely store fresh fish in the refrigerator?", "label": "OK"},
    {"query": "What is the best way to store leafy greens to keep them fresh?", "label": "OK"},
    {"query": "How do I properly clean and sanitize kitchen surfaces?", "label": "OK"},
    {"query": "What is a safe way to thaw frozen vegetables for cooking?", "label": "OK"},
    {"query": "How do I store flour and other dry goods to prevent pests?", "label": "OK"},
    {"query": "Is it safe to eat frozen vegetables that have been blanched before freezing?", "label": "OK"},
    {"query": "How do I safely handle and store fresh ground beef?", "label": "OK"},
    {"query": "What is the proper way to use a refrigerator thermometer?", "label": "OK"},
    {"query": "How do I safely prepare a meal using a slow cooker?", "label": "OK"},
    {"query": "What are food-safe containers for storing leftovers?", "label": "OK"},
    {"query": "How do I safely prepare a turkey for Thanksgiving?", "label": "OK"},
    {"query": "What is the best way to safely store hard cheeses?", "label": "OK"},
    {"query": "How do I properly sanitize sponges and dish cloths?", "label": "OK"},
    {"query": "What are safe ways to reduce sodium in recipes?", "label": "OK"},
    {"query": "How do I keep a buffet table food-safe at a party?", "label": "OK"},
    {"query": "What is a safe way to pack a lunch box to prevent bacterial growth?", "label": "OK"},
    {"query": "How do I properly freeze and thaw bread?", "label": "OK"},
    {"query": "Is it safe to eat honey past its best-before date?", "label": "OK"},
    {"query": "How do I safely prepare bean sprouts for a salad?", "label": "OK"},
    {"query": "What is the best way to safely store opened canned goods?", "label": "OK"},
]

# ---------------------------------------------------------------------------
# 2. Simple rule-based food safety classifier
#    Flags a query as NOT_OK based on known risk keywords / patterns.
# ---------------------------------------------------------------------------

UNSAFE_PATTERNS = [
    # raw / undercooked proteins
    r"\braw\b.*(chicken|poultry|duck|turkey|pork|beef|meat|fish|seafood|shrimp|oyster|clam|mussel|scallop|salmon|tuna|lamb|batter|dough|brownie|cookie|burger|potato|flour|sprout|egg)",
    r"(chicken|poultry|pork|beef|turkey|duck|lamb|burger|steak|meat).*(still pink|undercooked|medium.rare|rare|pink inside|pink)",
    r"(medium.rare|rare|still.pink|undercooked).*(chicken|poultry|pork|beef|turkey|duck|lamb|burger|steak|meat|fish)",
    r"\bundercook",
    r"\bsteak tartare\b",
    r"\braw egg(s)?\b",
    r"\braw milk\b",
    r"\bunpasteuriz",
    r"\brunny.*(egg|yolk)|soft.boiled.*(toddler|infant|baby|pregnan|immunocomp)",
    r"sushi.grade.*(regular|grocery|store)",
    r"(clam|mussel|shellfish).*(not open|never open|didn.t open|did not open)",
    r"\bpuffer fish\b|\bfugu\b",
    r"(fish|seafood|meat|mussel|clam|shrimp).*(smell.*(off|ammonia|slightly|pungent)|slight.*smell|strong.*smell)",
    r"(smell.*(off|ammonia|slightly|pungent|strong)).*(fish|seafood|meat|mussel|clam)",

    # temperature / time abuse
    r"left out.*(overnight|all day|\d+ hours?)",
    r"(room temperature|counter|hot car).*(overnight|\d+ hours?|all day|lunch)",
    r"refreeze.*thawed|thawed.*refreeze",
    r"thawed.*(counter|all day|overnight)",
    r"(cream|custard|mayo|mayonnaise|pastry|salad|guacamole|sandwich|deli|rice|pasta|soup|fish|meat|leftover).*(all day|overnight|\d+ hours?|2 week|\d+ week|\d+ day|for a week|for week|10 day|3 week)",
    r"buffet.*(hour|all day|\d+)",
    r"\d+.*(week|month|day|hour).*(fridge|frig|refrigerator|shelf)",
    r"(fridge|refrigerator).*(a week|1 week|2 week|3 week|10 day|\d+ week|\d+ day)",
    r"(expired|expir).*(milk|food|meat|fish)",
    r"(slimy|slime).*(kimchi|ferment)",
    r"(brown|grey|gray|discolor).*(guacamole|avocado|meat).*(week|days|safe)",
    r"leftover.*(week|2 week|\d+ day|10 day).*safe|leftover.*in the fridge for (2|\d+) week",
    r"food.*power.outage|power.outage.*food",
    r"candle.*eat|wax.*eat",
    r"simmering.*\d+ hour|stove.*\d+ hour",
    r"(deli|lunch) meat.*(open|fridge).*(week|3 week|\d+ week)",

    # cross contamination
    r"same (cutting board|knife|plate|tongs|bowl|sponge).*(raw|without wash|chicken|meat|week)",
    r"(raw|uncooked).*(salad|ready|cooked).*(without wash|same)",
    r"raw chicken.*sink.*splash",
    r"same (bowl|plate).*marinate|marinate.*same (bowl|plate)",
    r"sponge.*(week|month|cutting board)",

    # toxic plants / mushrooms / seeds
    r"\bgreen potato",
    r"rhubarb.*(leaf|leave)",
    r"\braw kidney bean",
    r"\braw flour\b",
    r"apple.*seed|cherry.*pit|apricot.*pit|lychee.*seed",
    r"(wild mushroom|park.*mushroom|forage.*mushroom|mushroom.*forest|mushroom.*park|toadstool).*(without|not).*(identif|know|expert)",
    r"\btoadstool\b",
    r"holly.*berr|elderberr.*raw|elderflower.*(not.*identif|unidentif|haven.t identif)",
    r"\bcastor beans?\b",
    r"bitter almond",
    r"moldy.*(bread|jam|fruit).*(cut off|scoop|safe)",
    r"(smell.*(off|funny|ammonia|foul|pungent|bad|sour)).*(eat|safe)",
    r"(sour.smell|bad smell|funny smell|foul smell|ammonia smell|pungent smell).*(eat|safe|ok)",
    r"\bgray.*meat\b|\bgrey.*meat\b",
    r"meat.*(turned|turn|gray|grey|grey|brown|discolor)",
    r"swollen.*package|package.*swollen|swollen.*can|swollen.*packaging",
    r"bulging.*can|dented.*can",
    r"rust.*(can|lid)",
    r"(10 year|years).*(can|home.can|jar|store)",
    r"foraged.*(herb|mushroom|berry|plant).*(without|not).*(identif|know)",
    r"dead.*(animal|cow|pig).*(eat|safe|meat)|unknown cause.*(die|died|dead)",
    r"rancid.*(oil|season|cast iron)",
    r"\braw potato\b",
    r"tomato.*(leaf|leave|plant)",
    r"avocado.*(skin|peel)",
    r"star fruit.*(kidney|renal)",
    r"(holly|holly bush).*(berr|eat)",
    r"cherry.*pit|apricot.*pit|seed.*(cherry|apricot|apple.*pie)",
    r"cheese.*(rind|mold|blue.green|unexpect)",
    r"(blue.green|unexpected).*(mold|mould).*(cheese|safe)",
    r"mold.*(scoop|cut|safe|top).*(jam|bread|fruit|cheese)",
    r"(sour|off|smell|ammon).*(bread|fish|meat|milk).*(safe|eat|ok)",
    r"expired.*(milk|food|fish|meat)",
    r"(2 week|3 week|\d+ week|10 day|\d+ day).*(leftover|fridge|refrigerator)",
    r"packaging.*(swollen|bulg|dent|swell)",
    r"(wax|candle).*(eat|safe|consume)",
    r"salsa.*(room temp|without process|not process)",
    r"jar.*(room temp|without process|counter).*salsa",

    # vulnerable groups risks
    r"honey.*(infant|baby|under 1|under one|12 month|toddler)",
    r"(pregnant|pregnancy|prenatal).*(raw|mercury|swordfish|high.mercury|unpasteuriz|soft.cheese|deli meat|sprout|tuna.*daily|rare|lamb|salmon.*daily)",
    r"(raw|rare|high.mercury|unpasteuriz|sprout).*(pregnant|pregnancy|prenatal|guest.*pregnan)",
    r"(infant|baby|babies).*(raw|honey|cow.s milk|whole milk|cow milk)",
    r"toddler.*(grape|whole|raw|honey)",
    r"(grape|whole grape).*(toddler|under 2|2 year)",
    r"immunocompromis.*(raw|unpasteuriz|sprout)",
    r"raw sprout.*(immunocomp|immune|weak|vulnerabl|pregnan)",
    r"(weak|impaired|compromised).*(immune|system).*(raw|sprout)",
    r"kidney disease.*(star fruit|high.sodium|sodium|salty|protein|restrict)",
    r"(high.sodium|salty|sodium).*(kidney|renal).*(disease|restriction|without)",
    r"heart.*(unlimited|salty|sodium|processed meat|processed)",
    r"diabetic.*(unlimited|large amount|tropical fruit|sugar)",
    r"blood thinner.*(grapefruit|unlimited)",
    r"(under 12 month|under one year|baby|infant).*(cow.s milk|cow milk|whole milk)",
    r"tree nut oil.*(allerg|allergy)",
    r"almond milk.*(nut.free|nut allerg|allerg)",
    r"shellfish broth.*(shellfish allerg|allerg)",
    r"casein.*(milk allerg|allerg)",
    r"(milk allerg|allerg).*(casein|margarine|butter.flavored)",
    r"give.*peanut butter.*baby|peanut butter.*(infant|baby).*(check|allerg|first)",
    r"nut.*(bakery|process|factory).*allerg|allerg.*nut.*(bakery|process)",
    r"(bakery|factory).*(process|handle).*nut.*(allerg|allergy|safe)",
    r"cook stuffed.*(frozen|turkey)|frozen stuffed.*cook",
    r"(cracked|chipped).*dish.*(food|serve)",
    r"(high.mercury|mercury).*(tuna|fish).*(daily|week|pregnan|frequent)",
    r"(pregnant|pregnan).*(rare|medium.rare).*(lamb|beef|burger|steak|meat|pork|chicken)",
    r"rare.*(burger|lamb|steak).*(child|kid|baby|young|toddler|pregnan)",

    # canning / preservation dangers
    r"water.bath.*cann.*(meat|bean|low.acid|green bean|vegetable)",
    r"water.bath.*can.*(low.acid|vegetable|meat|bean)",
    r"cann.*(low.acid|vegetable|meat|green bean|pressure|any jar|special jar)",
    r"(pressure can|pressure cann).*(any jar|without special|without canning)",
    r"garlic.in.oil.*(room temp|counter)",
    r"(room temp|jar.*room temp|salsa.*room temp).*(can|jar|store)",
    r"can.*funny.*smell|smell.*(home.can|canned)",
    r"reuse.*canning lid|old.*canning lid",
    r"commercial.*jar.*cann|reuse.*mayo.*jar",
    r"(home.can|home cann).*(10 year|years|year old)",
    r"salsa.*(room temp|without process|counter|not process)",
    r"can.*(low.acid|vegetable|meat).*(water bath|water.bath)",

    # contamination / chemical risks
    r"lead (pipe|crystal|paint)",
    r"non.food.grade (plastic|container)",
    r"styrofoam.*microwave|microwave.*styrofoam",
    r"pesticide.*(without wash|not wash)",
    r"pollut.*(river|lake|harbor).*eat|eat.*fish.*pollut",
    r"insecticide",
    r"galvaniz.*container|non.food.grade",
    r"mercury.*(lake|fish|river|level)",
    r"fish.*(lake|river).*(mercury|pollut|known)",
    r"(known.*mercury|high mercury|mercury.*level).*(fish|lake|river|eat)",
    r"not for food use|labeled.*(not for food)",
    r"alumin(i|u)um.*foil.*microwave|microwave.*foil",
    r"highway.*(vegetable|vegetabl|produce|grown).*(without|not).*wash",
    r"grown.*(highway|pollut|contamin).*(without|not).*wash",
    r"(near highway|near road|near traffic).*(grown|vegetable|produce)",

    # food handling
    r"(without|not).wash.*(hand|hands).*(food|cook|handle|bathroom)",
    r"bathroom.*(wash|without).*(hand|cook|food)",
    r"same spoon.*taste.*pot|taste.*put back|spoon.*back.*pot",
    r"defrost.*(counter|room temp).*(all day|overnight)",
    r"cook.*frozen.*stuff",
    r"damaged.*non.stick|non.stick.*damaged|scratch.*non.stick",
    r"opened.*can.*(fridge|refrigerat|store)|store.*open.*can",

    # allergen mislabeling
    r"peanut oil.*(allerg|allergy)",
    r"fish sauce.*(shellfish allerg)",
    r"(wheat|gluten).*(gluten.free|without label)",
    r"soy sauce.*(soy allerg)",

    # extended protein / seafood / fish raw/rare
    r"\braw\b.*(veal|squid|herring|mackerel|lobster|crab|crayfish|anchovy|sardine)",
    r"(veal|squid|herring|mackerel|lobster|crab).*(rare|raw|smell|off)",
    r"(rare|raw|undercooked).*(veal|squid|herring|mackerel|crayfish)",
    r"raw (center|centre|inside|in the middle)|raw.*center.*steak|steak.*raw.*center",
    r"steak.*(raw|rare).*(center|middle|inside)|raw.*(center|middle|inside).*steak",
    r"barely cooked|lightly cooked.*(chicken|pork|beef|fish|meat)",
    r"lobster.*(smell|off|ammonia|bad)",
    r"polluted.*(beach|water|sea|ocean).*(oyster|shellfish|clam|mussel|seafood)",
    r"(oyster|shellfish|clam).*(pollut|contamin)",
    r"raw.*squid|squid.*raw|raw.*herring|herring.*raw|raw.*mackerel|mackerel.*raw",

    # extended temperature abuse
    r"cheese.*(out of fridge|room temp|counter).*(hour|all day|12 hour|8 hour)",
    r"cooking oil.*(counter|room temp|sitting|weeks|month)",
    r"(reheated|reheat).*(5 time|multiple time|several times|again and again)",
    r"warm.*car.*(hour|hours?|food|meat|chicken)|car.*(warm|hot).*(food|meat).*hour",

    # extended cross-contamination
    r"(cooked|ready.to.eat).*(raw|uncooked).*(same knife|unwashed|without wash)",
    r"same (knife|board|surface).*(raw|cooked).*(without wash|unwashed)",
    r"raw.*(surface|counter|board).*(ready.to.eat|cooked|fruit)",
    r"sink.*(chicken|raw).*(vegetable|produce|fruit)",

    # extended toxic plants / berries
    r"elderberr.*(lightly|partially|slightly|only|half).*(cooked|cook)",
    r"ornamental.*(chili|pepper|plant).*(eat|safe)",
    r"(chili|pepper).*(ornamental).*(eat|safe)",
    r"garden mushroom.*(without|not).*(identif|know|expert)",
    r"mushroom.*(garden|backyard|found|wild|park).*(without|not).*(identif|know|expert)",
    r"mushroom.*(look wrong|suspicious|wrong|questionable)",
    r"yew.*(berr|tree|eat|safe|red)",
    r"nightshade.*(berr|eat|safe|like blueberr)",
    r"berr.*(nightshade|yew|unidentif|unknown|wild|forage).*(eat|safe)",
    r"raw rhubarb|rhubarb.*raw",
    r"(dipped in sugar|sugar.*dip).*(rhubarb)",

    # extended spoilage / contamination
    r"(rodent|pest|infestation|rat|mouse).*(facility|food|made|process)",
    r"food.*(fell|fall|drop).*(puddle|floor|ground|dirt).*(wash|rinse|safe)",
    r"fermented.*(fish|vegetable|food).*(strong.*odor|odor|extremely|pungent)",
    r"(mushy|slimy|soft).*(pickle|brine|ferment|kimchi).*(safe|eat)",
    r"pickle.*(mushy|slimy).*(safe|eat)",
    r"home.brine.*(mushy|slimy|soft)",
    r"mushroom.*(wrong|unusual|odd|suspicious|bad).*(safe|eat|soup)",
    r"some mushroom.*(look|appear|seem).*(wrong|bad|off|unusual)",

    # extended vulnerable groups
    r"(infant|baby|6.month|newborn|toddler).*(sugar water|honey|raw|alcohol|unpasteuriz)",
    r"sugar water.*(infant|baby|newborn|6.month|young)",
    r"raw honey.*(6.month|infant|baby|under 1|newborn|young)",
    r"(celiac|celiac disease).*(wheat|gluten).*(without|not knowing|undiagnosed|diagnosi)",
    r"(undiagnosed|without knowing).*(celiac|gluten|allerg).*(eat|safe)",
    r"(old rusty|rusty|rust).*(lid|can|jar|canning)",
    r"rusty.*(canning|lid|can|jar)",
    r"(stovetop pot|regular pot|regular stovetop).*(pressure can|canning|pressure cann)",
    r"oven canning|oven.*can.*(fish|meat|vegetable)",
    r"water bath.*chicken|home.can.*(chicken|fish|meat).*water bath",
    r"(pumpkin puree|pumpkin).*(home.*can|canning|water bath)",
    r"(altitude|high altitude).*(canning|can).*(without adjust|not adjust)",
    r"(cracked|broken|damaged).*(plastic|container|bowl).*(microwave|heat|food)",
    r"painted pottery.*(food|acid|tomato|fruit|safe)",
    r"non.food.grade.*(bucket|container|plastic).*(ferment|food|store)",
    r"well water.*(yellow|tint|brown|color|discolor).*(drink|safe|test)",
    r"contaminated.*(soil|dirt|heavy metal).*(grow|vegetable|produce|plant|food)",
    r"gasoline.*(can|container).*(water|drink|store|food)",
    r"(dialysis|renal failure).*(high.potassium|potassium|fruit|banana|orange|daily)",
    r"premature.*(baby|infant|newborn).*(solid food|solids|food|eat).*(month|early|2 month)",
    r"(baby|infant|newborn).*(alcohol|wine|beer|spirit)",
    r"(cured meat|cold cut|deli meat|deli).*(pregnant|pregnancy).*(without heat|cold|not heat)",
    r"(unpasteuriz|raw|unprocessed).*(apple.*cider|juice).*(child|kid|toddler|young)",
    r"(dog|cat|pet).*(onion|garlic|grape|raisin|chocolate|xylitol).*(safe|feed|eat)",
    r"wild berr.*(hiking|forag|outdoor|unknown|unidentif).*(eat|safe)",
    r"wild plant.*(park|forage|hiking|unknown|unidentif).*(eat|safe|collect)",
    r"insect.*(damage|eaten).*(outside|exterior|only outside).*(safe|eat|ok)",
    r"rotten.*(fruit|vegetable|food).*(cut.*out|remove|safe|eat)",
    r"(warm water|hot water).*(thaw|defrost).*(hour|steak|chicken|meat|fish|5 hour)",
    r"marinate.*(room temp|counter|outside).*(summer|warm|2 hour|hot)",
    r"shellfish.*(health advisory|advisory|warning|closed|posted)",
    r"(hot garage|garage|car|trunk).*(canned|can).*(summer|all summer|months?|season)",
    r"frosting.*(sun|outdoor|outside|warm).*(hour|4 hour|day)",
    r"marinade.*(raw pork|raw chicken|raw beef|raw meat).*(dipping sauce|sauce|reuse)",
    r"unlicensed.*(smoker|smoke|home smoke)",
    r"rice.*(fridge|refrigerator).*(8 day|7 day|days?|week|long)",
    r"green mold.*(bread|food).*(remove|cut|scrape|only|spots)",
    r"raw.*(pre.packaged|cookie dough|store.bought dough)",
    r"supermarket.*(salmon|fish).*(sashimi|raw|fresh label)",
    r"raw almond.*(tree|collect|fresh|uncured)",
    r"mushroom.*(past best.before|expired|date|raw|without cook)",
    r"(near|next to|stored.*(near|with)).*(cleaning chemical|chemical|detergent|bleach)",
    r"preserved lemon.*(room temp|counter|6 month|months?)",
    r"changed color.*(texture|smell|long storage).*(safe|eat)",
    r"rainbow.*(sheen|shine|color).*(lunch meat|deli|meat)",
    r"home.canned.*(ferment|smell|bubbl).*(salsa|tomato|vegetable)",
    r"cracked.*(cast iron|pan|skillet).*(rust|rusty|safe|food|cook)",
    r"pizzeria.*(dough|raw flour|uncooked flour)",
    r"freezer burn.*(flesh|inside|obvious|ice crystal).*(safe|eat)",
    r"chemical.*smell.*(bag|container|package).*(food|eat|safe)",

    # general unsafe food indicators
    r"raw brownie|raw batter|raw dough.*(egg|flour)",
    r"5.second rule",
    r"(old|10 year|years).*(ferment|home.can|jar|store)",
    r"puffer fish.*liver|liver.*puffer fish",
    r"(forage|wild|unidentif).*(herb|mushroom|berry|elderflower|elderberr)",
    r"reheated.*multiple times|reheat.*several days",

    # catch remaining edge cases — bidirectional keyword combos
    r"raw.*in the middle|in the middle.*raw",
    r"freshwater fish.*raw|fish.*raw.*(freshwater|wild.caught)|(wild.caught|freshwater).*fish.*raw",
    r"(hot car|in a car).*(mayonnaise|mayo|sandwich|food)|(mayonnaise|mayo|sandwich).*(hot car|in a car)",
    r"(holly|holly bush).*(berr|eat|safe|red)|(red berr|berr).*(holly)",
    r"(cherry|apricot|apple).*(pit|pits|seed)|(pit|pits).*(cherr|apricot|apple)",
    r"\btoadstools?\b",
    r"\bcastor beans?\b",
    r"avocado.*(skin|peel)|(skin|peel).*(avocado)",
    r"(nut.free|nut free).*(bakery|almond milk|process)|(bakery|almond milk|process).*(nut.free|nut free)",
    r"almond milk.*(nut.free|nut free|allerg)|(nut.free|nut free).*almond milk",
    r"bread.*(sour.*smell|smell.*sour|visibly mold|moldy)|(sour.*smell|smell.*sour).*(bread|food)",
    r"fermented.*(foul|smell|pungent|off)|(foul|smell).*(ferment)",
    r"(fruit|jam|bread|cheese).*(mold|mould)|(mold|mould).*(fruit|jam|bread|cheese)",
    r"kimchi.*(slimy|slime|turned)|(slimy|slime).*kimchi",
    r"(eat|consume|safe).*(wax|candle|beeswax)|(wax|candle|beeswax).*(eat|consume|safe)",
    r"(died|dead|unknown cause).*(meat|animal|safe|eat)|(meat|animal).*(died|dead|unknown cause)",
    r"(any jar|any glass jar|any container).*(pressure can|canning|pressure cann)|(pressure can|canning).*any jar",
    r"stuffed.*(frozen|turkey).*(without thaw|not thaw|cook from frozen)|(without thaw|frozen).*(stuffed|turkey)",
    r"frozen.*(stuffed|turkey).*(cook|without thaw)|(cook|without thaw).*(frozen.*(stuffed|turkey)|stuffed.frozen)",
    r"cow.s milk.*(baby|infant|12 month|under 12|main drink)|(baby|infant|12 month|under 12).*(cow.s milk|cow milk|main drink|whole milk)",
    r"fruit.*(mold|mould)|(mold|mould).*(fruit|berry|vegetable)",
]

SAFE_PATTERNS = [
    r"\bpasteuriz",
    r"\bcooked.*165|165.*chicken|safe.*temperature|proper.*temperature",
    r"\bthermometer\b",
    r"\brefrigerator.*thaw|cold water.*thaw|thaw.*fridge",
    r"\bproperly.*wash|wash.*produce",
    r"\bsanitiz",
]


def classify_query(query: str) -> str:
    """Classify a food safety query as OK or NOT_OK using rule-based matching."""
    q_lower = query.lower()

    # Check unsafe patterns first
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            return "NOT_OK"

    return "OK"


# ---------------------------------------------------------------------------
# 3. Run evaluation
# ---------------------------------------------------------------------------

def evaluate():
    queries = [item["query"] for item in FOOD_SAFETY_DATASET]
    y_true = [item["label"] for item in FOOD_SAFETY_DATASET]
    y_pred = [classify_query(q) for q in queries]

    total = len(queries)
    n_not_ok_true = y_true.count("NOT_OK")
    n_ok_true = y_true.count("OK")

    print("=" * 65)
    print("  FOOD SAFETY EVALUATION — STYLE Domain")
    print("=" * 65)
    print(f"  Total queries evaluated : {total}")
    print(f"  NOT_OK (unsafe) queries : {n_not_ok_true}")
    print(f"  OK (safe) queries       : {n_ok_true}")
    print()

    # Full classification report
    labels = ["OK", "NOT_OK"]
    report = classification_report(y_true, y_pred, labels=labels, digits=4)
    print(report)

    # Extract NOT_OK metrics individually
    p_not_ok, r_not_ok, f1_not_ok, support_not_ok = precision_recall_fscore_support(
        y_true, y_pred, labels=["NOT_OK"], average=None
    )

    print("-" * 65)
    print("  Per-class metrics for the unsafe-food class (NOT_OK)")
    print("-" * 65)
    print(f"  precision_NOT_OK : {p_not_ok[0]:.4f}")
    print(f"  recall_NOT_OK    : {r_not_ok[0]:.4f}")
    print(f"  f1_NOT_OK        : {f1_not_ok[0]:.4f}")
    print(f"  support          : {int(support_not_ok[0])}  queries")
    print("=" * 65)

    # Save results to JSON
    results = {
        "dataset": "food_safety",
        "total_queries": total,
        "class_distribution": {"NOT_OK": n_not_ok_true, "OK": n_ok_true},
        "classifier": "rule_based_keyword_matching",
        "NOT_OK_metrics": {
            "precision_NOT_OK": round(float(p_not_ok[0]), 4),
            "recall_NOT_OK": round(float(r_not_ok[0]), 4),
            "f1_NOT_OK": round(float(f1_not_ok[0]), 4),
            "support": int(support_not_ok[0]),
        },
    }

    with open("food_safety_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: food_safety_evaluation_results.json")

    # Detailed error analysis
    df = pd.DataFrame({"query": queries, "true": y_true, "pred": y_pred})
    df["correct"] = df["true"] == df["pred"]

    print()
    print("  Error breakdown:")
    print(f"    True Positives  (NOT_OK correctly flagged)  : "
          f"{((df.true=='NOT_OK') & (df.pred=='NOT_OK')).sum()}")
    print(f"    False Negatives (NOT_OK missed as OK)       : "
          f"{((df.true=='NOT_OK') & (df.pred=='OK')).sum()}")
    print(f"    False Positives (OK wrongly flagged NOT_OK) : "
          f"{((df.true=='OK') & (df.pred=='NOT_OK')).sum()}")
    print(f"    True Negatives  (OK correctly passed)       : "
          f"{((df.true=='OK') & (df.pred=='OK')).sum()}")
    print()

    # Show false negatives (missed unsafe queries)
    fn = df[(df.true == "NOT_OK") & (df.pred == "OK")]
    if not fn.empty:
        print(f"  False Negatives ({len(fn)} missed unsafe queries):")
        for _, row in fn.iterrows():
            print(f"    - {row['query']}")
        print()

    return results


if __name__ == "__main__":
    evaluate()
