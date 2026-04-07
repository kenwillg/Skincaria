"""
generate_dataset.py
Generates instruction-data.json for Chapter 7 LLM fine-tuning.
Combines 4 data sources into conversational skincare recommendation data.
"""

import pandas as pd
import json
import random
import re
import os

random.seed(42)

# ============================================================================
# 1. LOAD ALL DATASETS
# ============================================================================
BASE_SOCIOLLA = "dataset/Indonesian Skincare Sample Dataset"
BASE_INCI = "dataset/Skin care product ingredients - INCI List"

print("Loading datasets...")
products = pd.read_excel(f"{BASE_SOCIOLLA}/product.xlsx")
ingredients_cat = pd.read_excel(f"{BASE_SOCIOLLA}/ingredients_category.xlsx")
claims = pd.read_excel(f"{BASE_SOCIOLLA}/product_claim_category.xlsx")
inci = pd.read_csv(f"{BASE_INCI}/ingredientsList.csv")

print(f"  Products: {len(products)} rows")
print(f"  Ingredients Category: {len(ingredients_cat)} rows")
print(f"  Product Claims: {len(claims)} rows")
print(f"  INCI List: {len(inci)} rows")

# ============================================================================
# 2. BUILD LOOKUP DICTIONARIES
# ============================================================================
print("\nBuilding lookup dictionaries...")

# --- Claims translation: Indonesian -> English ---
claims_map = {}
for _, row in claims.iterrows():
    indo_desc = str(row["description_product"]).strip()
    eng_cat = str(row["claim_category"]).strip()
    claims_map[indo_desc] = eng_cat

# --- Ingredients Category lookup (by normalized name) ---
ingr_cat_lookup = {}
for _, row in ingredients_cat.iterrows():
    name = str(row["ingredient_name"]).strip().lower()
    ingr_cat_lookup[name] = {
        "function1": str(row.get("function1", "")).strip() if pd.notna(row.get("function1")) else None,
        "function2": str(row.get("function2", "")).strip() if pd.notna(row.get("function2")) else None,
        "warning1": str(row.get("warning1", "")).strip() if pd.notna(row.get("warning1")) else None,
        "warning2": str(row.get("warning2", "")).strip() if pd.notna(row.get("warning2")) else None,
        "origin": str(row.get("ingredient_origin", "")).strip() if pd.notna(row.get("ingredient_origin")) else None,
    }

# --- INCI lookup (by normalized name) ---
inci_lookup = {}
for _, row in inci.iterrows():
    if pd.isna(row.get("name")):
        continue
    name = str(row["name"]).strip().lower()

    # Parse who_is_it_good_for -> clean list
    good_for_raw = str(row.get("who_is_it_good_for", "[]"))
    good_for = [x.strip().strip("'\"") for x in re.findall(r"'([^']*)'", good_for_raw)]
    good_for = [x for x in good_for if x.strip() and x.strip() != ""]

    # Parse who_should_avoid -> clean list
    avoid_raw = str(row.get("who_should_avoid", "[]"))
    avoid = [x.strip().strip("'\"") for x in re.findall(r"'([^']*)'", avoid_raw)]
    avoid = [x for x in avoid if x.strip() and x.strip() != ""]

    inci_lookup[name] = {
        "what_is_it": str(row.get("what_is_it", "")).strip() if pd.notna(row.get("what_is_it")) else None,
        "what_does_it_do": str(row.get("what_does_it_do", "")).strip() if pd.notna(row.get("what_does_it_do")) else None,
        "good_for": good_for,
        "avoid": avoid,
    }

print(f"  Claims map: {len(claims_map)} translations")
print(f"  Ingredient category lookup: {len(ingr_cat_lookup)} ingredients")
print(f"  INCI lookup: {len(inci_lookup)} ingredients")

# ============================================================================
# 3. PROFILE EACH PRODUCT
# ============================================================================
print("\nProfiling products...")


def parse_ingredients(ingredients_str):
    """Parse comma-separated ingredient list from product."""
    if pd.isna(ingredients_str):
        return []
    return [x.strip() for x in str(ingredients_str).split(",") if x.strip()]


def match_ingredient(name):
    """Try to match an ingredient name to INCI and category lookups."""
    normalized = name.strip().lower()
    # Try exact match
    inci_info = inci_lookup.get(normalized)
    cat_info = ingr_cat_lookup.get(normalized)

    # Try without parenthetical (e.g., "Aqua (Water)" -> "aqua")
    if not inci_info:
        base_name = re.sub(r'\s*\([^)]*\)', '', normalized).strip()
        inci_info = inci_lookup.get(base_name)
    if not cat_info:
        base_name = re.sub(r'\s*\([^)]*\)', '', normalized).strip()
        cat_info = ingr_cat_lookup.get(base_name)

    return inci_info, cat_info


def translate_claims(description_product):
    """Translate Indonesian product description to English claim categories."""
    if pd.isna(description_product):
        return []
    # Split by comma or newline
    parts = re.split(r'[,\\n]+', str(description_product))
    english_claims = []
    for part in parts:
        part = part.strip().rstrip("\\n").strip()
        if part in claims_map:
            english_claims.append(claims_map[part])
        elif part:
            # Try partial matching
            for indo_key, eng_val in claims_map.items():
                if indo_key in part or part in indo_key:
                    english_claims.append(eng_val)
                    break
    return list(set(english_claims))  # deduplicate


product_profiles = []
for _, row in products.iterrows():
    ingredients = parse_ingredients(row.get("ingredients_list"))
    eng_claims = translate_claims(row.get("description_product"))

    # Collect info from matched ingredients
    good_for_set = set()
    avoid_set = set()
    warnings = set()
    functions = set()
    key_ingredients = []  # Ingredients with rich INCI data

    for ingr_name in ingredients:
        inci_info, cat_info = match_ingredient(ingr_name)

        if inci_info:
            for concern in inci_info.get("good_for", []):
                good_for_set.add(concern)
            for concern in inci_info.get("avoid", []):
                avoid_set.add(concern)
            key_ingredients.append({
                "name": ingr_name,
                "what_does_it_do": inci_info.get("what_does_it_do"),
                "what_is_it": inci_info.get("what_is_it"),
                "good_for": inci_info.get("good_for", []),
                "avoid": inci_info.get("avoid", []),
            })

        if cat_info:
            if cat_info.get("function1"):
                functions.add(cat_info["function1"])
            if cat_info.get("function2"):
                functions.add(cat_info["function2"])
            if cat_info.get("warning1"):
                warnings.add(cat_info["warning1"])
            if cat_info.get("warning2"):
                warnings.add(cat_info["warning2"])

    profile = {
        "brand": str(row.get("brand", "")).strip(),
        "product_name": str(row.get("product_name", "")).strip(),
        "product_type": str(row.get("product_type", "")).strip(),
        "size": str(row.get("size", "")).strip(),
        "normal_price": row.get("normal_price", 0),
        "discount_price": row.get("discount_price", 0),
        "discount": row.get("discount", 0),
        "rating": row.get("rating", 0),
        "review_count": row.get("review_count", 0),
        "english_claims": eng_claims,
        "good_for": sorted(good_for_set),
        "avoid": sorted(avoid_set),
        "warnings": sorted(warnings),
        "functions": sorted(functions),
        "key_ingredients": key_ingredients[:8],  # Top 8 with INCI data
        "all_ingredients": ingredients,
    }
    product_profiles.append(profile)

matched_counts = [len(p["key_ingredients"]) for p in product_profiles]
print(f"  Profiled {len(product_profiles)} products")
print(f"  Avg matched INCI ingredients per product: {sum(matched_counts)/len(matched_counts):.1f}")
print(f"  Products with claims: {sum(1 for p in product_profiles if p['english_claims'])}")

# ============================================================================
# 4. CONVERSATION TEMPLATES
# ============================================================================

# Skin concern synonyms for diverse user inputs
SKIN_CONCERNS = {
    "Acne": [
        "I have acne and frequent breakouts.",
        "My face is covered in pimples and acne.",
        "I struggle with persistent acne on my face.",
        "I keep getting acne breakouts, especially on my chin and forehead.",
        "My skin is acne-prone and I get blemishes often.",
    ],
    "Blackheads": [
        "I have a lot of blackheads on my nose.",
        "My pores are clogged with blackheads.",
        "I notice blackheads forming on my T-zone area.",
        "I struggle with stubborn blackheads that keep coming back.",
    ],
    "Redness": [
        "My skin gets red and irritated easily.",
        "I have redness on my cheeks that won't go away.",
        "My face is always flushed and red.",
        "I deal with persistent redness and skin irritation.",
    ],
    "Dry and dehydrated skin": [
        "My skin feels very dry and tight.",
        "My skin is extremely dehydrated and flaky.",
        "I have dry patches on my face that feel uncomfortable.",
        "My skin lacks moisture and feels rough to the touch.",
        "My face always feels parched and tight, especially after washing.",
    ],
    "Impaired skin barrier": [
        "My skin barrier is damaged and everything stings.",
        "I think my skin barrier is compromised because products burn when I apply them.",
        "My skin is very reactive and I believe my skin barrier is impaired.",
        "My skin has become very sensitive recently and feels like the barrier is broken.",
    ],
    "Fine Lines": [
        "I'm starting to notice fine lines around my eyes.",
        "I have fine lines and early signs of aging.",
        "I want to reduce the appearance of fine lines on my forehead.",
        "I'm concerned about the fine lines forming on my face.",
    ],
    "Dullness": [
        "My skin looks dull and lifeless.",
        "My complexion is very dull and lacks radiance.",
        "My face looks tired and lacks any glow.",
        "I want my skin to look more radiant and less dull.",
    ],
    "Oily skin": [
        "My skin gets very oily by midday.",
        "I have extremely oily skin, especially in the T-zone.",
        "My face is always shiny and greasy.",
        "I produce too much oil on my face throughout the day.",
    ],
    "Dark spots": [
        "I have dark spots and hyperpigmentation on my cheeks.",
        "I want to fade the dark spots left by old acne scars.",
        "My skin has uneven tone with dark patches.",
        "I struggle with post-inflammatory hyperpigmentation.",
    ],
    "Radiance": [
        "I want my skin to look more glowing and radiant.",
        "I'm looking for products to brighten my complexion.",
        "I want a healthy, radiant glow on my skin.",
    ],
    "Texture": [
        "My skin texture is very rough and uneven.",
        "I want smoother skin with a better texture.",
        "My skin feels bumpy and the texture is uneven.",
    ],
    "Wrinkles": [
        "I have noticeable wrinkles and want to reduce them.",
        "I'm concerned about deep wrinkles forming on my face.",
        "I need anti-aging products for my wrinkles.",
    ],
    "Pregnancy": [
        "I am pregnant and looking for safe skincare products.",
        "I need pregnancy-safe skincare recommendations.",
        "What products are safe to use during pregnancy?",
    ],
    "Elasticity": [
        "My skin has lost its firmness and elasticity.",
        "I want products that help with skin elasticity and firmness.",
        "My skin feels saggy and needs more elasticity.",
    ],
}

# Instruction templates
RECOMMENDATION_INSTRUCTIONS = [
    "I need a skincare product recommendation.",
    "Can you recommend a skincare product for me?",
    "What skincare product should I use?",
    "Please suggest a product for my skin concerns.",
    "Help me find the right skincare product.",
    "I'm looking for a skincare recommendation.",
    "What would you recommend for my skin?",
    "Can you help me choose a skincare product?",
    "Suggest a good skincare product for me.",
    "I need help choosing a skincare product.",
]

INQUIRY_INSTRUCTIONS = [
    "Is this product suitable for my skin?",
    "Can you tell me if this product is right for me?",
    "Would this product work for my skin type?",
    "Should I use this product given my skin concerns?",
    "Is this product safe for me to use?",
    "Do you think this product would be good for my skin?",
]

INGREDIENT_INSTRUCTIONS = [
    "Tell me about a skincare ingredient.",
    "What can you tell me about this skincare ingredient?",
    "I want to learn about a skincare ingredient.",
    "Explain what this ingredient does in skincare.",
    "What is this skincare ingredient used for?",
    "Can you explain this ingredient to me?",
]

CLAIMS_INSTRUCTIONS = [
    "What are the benefits of this product?",
    "What does this product do for the skin?",
    "Can you tell me what this product claims to do?",
    "What skin benefits does this product offer?",
    "Tell me about the key benefits of this product.",
]

AVOIDANCE_INSTRUCTIONS = [
    "What skincare ingredients should I avoid?",
    "Are there any ingredients I should stay away from?",
    "Which ingredients could be harmful for my skin?",
    "What should I avoid in skincare products?",
    "Tell me about ingredients to avoid for my condition.",
]


# ============================================================================
# 5. GENERATE CONVERSATIONS
# ============================================================================
print("\nGenerating conversations...")
dataset = []


def format_price(price):
    """Format price in Indonesian Rupiah."""
    try:
        return f"Rp {int(price):,}".replace(",", ".")
    except (ValueError, TypeError):
        return "price not available"


# --- TYPE 1: Symptom -> Product Recommendation (~200) ---
print("  Type 1: Symptom -> Recommendation...")
type1_count = 0
for concern, user_phrases in SKIN_CONCERNS.items():
    # Find products good for this concern
    matching_products = [p for p in product_profiles if concern in p["good_for"]]
    if not matching_products:
        continue

    for product in matching_products:
        # Find the key ingredient that matches this concern
        relevant_ingrs = [
            ki for ki in product["key_ingredients"]
            if concern in ki.get("good_for", [])
        ]
        if not relevant_ingrs:
            continue

        user_input = random.choice(user_phrases)
        instruction = random.choice(RECOMMENDATION_INSTRUCTIONS)

        # Build the response
        ingr = relevant_ingrs[0]
        ingr_explanation = ingr.get("what_does_it_do", "")
        if ingr_explanation and len(ingr_explanation) > 300:
            # Trim to first two sentences
            sentences = ingr_explanation.split(".")
            ingr_explanation = ". ".join(sentences[:2]) + "."

        response_parts = [
            f"For your concern with {concern.lower()}, I recommend the {product['brand']} {product['product_name']}.",
            f"It is a {product['product_type']} that contains {ingr['name']}.",
        ]
        if ingr_explanation:
            response_parts.append(f"{ingr_explanation}")
        if product["rating"] > 0:
            response_parts.append(
                f"This product is rated {product['rating']} stars with {product['review_count']} reviews"
                f" and is priced at {format_price(product['discount_price'])}."
            )

        # Add warning about things to avoid if applicable
        if product["warnings"]:
            response_parts.append(
                f"Please note that this product contains ingredients flagged as: {', '.join(product['warnings'])}."
                f" If you have sensitive skin, do a patch test first."
            )

        entry = {
            "instruction": instruction,
            "input": user_input,
            "output": " ".join(response_parts),
        }
        dataset.append(entry)
        type1_count += 1

# Add some multi-concern recommendations
for _ in range(50):
    # Pick 2 random concerns
    concerns = random.sample([c for c in SKIN_CONCERNS.keys() if c != "Pregnancy"], min(2, len(SKIN_CONCERNS)))
    matching = [
        p for p in product_profiles
        if all(c in p["good_for"] for c in concerns)
    ]
    if not matching:
        continue

    product = random.choice(matching)
    user_input = " Also, ".join([
        random.choice(SKIN_CONCERNS[c]) for c in concerns
    ])
    instruction = random.choice(RECOMMENDATION_INSTRUCTIONS)

    # Find relevant ingredients for the concerns
    relevant_ingrs = []
    for ki in product["key_ingredients"]:
        for c in concerns:
            if c in ki.get("good_for", []):
                relevant_ingrs.append(ki)
                break
    relevant_ingrs = relevant_ingrs[:3]

    response = f"Based on your concerns with {' and '.join([c.lower() for c in concerns])}, I recommend the {product['brand']} {product['product_name']}. "
    response += f"This {product['product_type']} is great because it contains: "
    for ri in relevant_ingrs:
        desc = ri.get("what_does_it_do", "")
        if desc and len(desc) > 150:
            desc = ". ".join(desc.split(".")[:1]) + "."
        response += f"{ri['name']}, which {desc.lower() if desc else 'supports skin health'}. "
    response += f"It is priced at {format_price(product['discount_price'])} and rated {product['rating']} stars."

    dataset.append({
        "instruction": instruction,
        "input": user_input,
        "output": response,
    })
    type1_count += 1

print(f"    Generated {type1_count} entries")

# --- TYPE 2: Product Inquiry (Is this good for me?) (~100) ---
print("  Type 2: Product Inquiry...")
type2_count = 0
for product in product_profiles:
    if not product["key_ingredients"]:
        continue

    # Positive inquiry (product IS good for the user's concern)
    if product["good_for"]:
        concern = random.choice(product["good_for"])
        if concern in SKIN_CONCERNS:
            user_input = f"I have {concern.lower()}. Is the {product['brand']} {product['product_name']} good for me?"
            instruction = random.choice(INQUIRY_INSTRUCTIONS)

            relevant_ingr = None
            for ki in product["key_ingredients"]:
                if concern in ki.get("good_for", []):
                    relevant_ingr = ki
                    break

            response = f"Yes, the {product['brand']} {product['product_name']} is a great choice for {concern.lower()}! "
            if relevant_ingr:
                desc = relevant_ingr.get("what_does_it_do", "")
                if desc and len(desc) > 200:
                    desc = ". ".join(desc.split(".")[:2]) + "."
                response += f"It contains {relevant_ingr['name']}, which {desc.lower() if desc else 'helps with your skin concern'}. "
            if product["english_claims"]:
                response += f"Its key benefits include: {', '.join(product['english_claims'][:4])}. "
            response += f"It is rated {product['rating']} stars and priced at {format_price(product['discount_price'])}."

            dataset.append({
                "instruction": instruction,
                "input": user_input,
                "output": response,
            })
            type2_count += 1

    # Negative inquiry (product has warnings relevant to the user)
    if product["warnings"] and product["avoid"]:
        avoid_concern = random.choice(product["avoid"])
        if "Allergy" in avoid_concern:
            avoid_concern = "sensitive skin"

        user_input = f"I have {avoid_concern.lower()}. Should I use the {product['brand']} {product['product_name']}?"
        instruction = random.choice(INQUIRY_INSTRUCTIONS)

        response = f"You should be cautious with the {product['brand']} {product['product_name']}. "
        response += f"While it is rated {product['rating']} stars, it contains ingredients flagged with the following warnings: {', '.join(product['warnings'])}. "
        response += f"These may not be suitable for {avoid_concern.lower()}. "
        response += "I recommend doing a patch test before using this product, or considering an alternative that is gentler on your skin."

        dataset.append({
            "instruction": instruction,
            "input": user_input,
            "output": response,
        })
        type2_count += 1

print(f"    Generated {type2_count} entries")

# --- TYPE 3: Ingredient Education (~150) ---
print("  Type 3: Ingredient Education...")
type3_count = 0
for ingr_name, info in inci_lookup.items():
    what_is_it = info.get("what_is_it")
    what_does = info.get("what_does_it_do")
    good_for = info.get("good_for", [])
    avoid = info.get("avoid", [])

    if not what_is_it and not what_does:
        continue

    display_name = ingr_name.title()
    instruction = random.choice(INGREDIENT_INSTRUCTIONS)

    # Type 3a: "What is X?"
    if what_is_it:
        user_input = f"What is {display_name} and what does it do for the skin?"
        response = f"{display_name} is a skincare ingredient. {what_is_it} "
        if what_does:
            does_text = what_does
            if len(does_text) > 300:
                does_text = ". ".join(does_text.split(".")[:3]) + "."
            response += f"{does_text} "
        if good_for:
            response += f"It is particularly good for: {', '.join(good_for)}. "
        if avoid:
            response += f"People with {', '.join(avoid).lower()} should be cautious when using this ingredient."

        dataset.append({
            "instruction": instruction,
            "input": user_input,
            "output": response.strip(),
        })
        type3_count += 1

    # Type 3b: "Is X good for [concern]?"
    if good_for:
        concern = random.choice(good_for)
        instruction2 = random.choice(INGREDIENT_INSTRUCTIONS)
        user_input2 = f"Is {display_name} good for {concern.lower()}?"

        response2 = f"Yes, {display_name} is beneficial for {concern.lower()}. "
        if what_does:
            does_text = what_does
            if len(does_text) > 200:
                does_text = ". ".join(does_text.split(".")[:2]) + "."
            response2 += f"{does_text} "
        other_benefits = [g for g in good_for if g != concern]
        if other_benefits:
            response2 += f"It is also good for: {', '.join(other_benefits[:3])}."

        dataset.append({
            "instruction": instruction2,
            "input": user_input2,
            "output": response2.strip(),
        })
        type3_count += 1

print(f"    Generated {type3_count} entries")

# --- TYPE 4: Product Claims (What does this product do?) (~80) ---
print("  Type 4: Product Claims...")
type4_count = 0
for product in product_profiles:
    if not product["english_claims"]:
        continue

    instruction = random.choice(CLAIMS_INSTRUCTIONS)
    user_input = f"What does the {product['brand']} {product['product_name']} do?"

    response = f"The {product['brand']} {product['product_name']} is a {product['product_type']} that offers the following benefits: "
    response += f"{', '.join(product['english_claims'])}. "

    if product["key_ingredients"]:
        top_ingrs = product["key_ingredients"][:3]
        ingr_names = [ki["name"] for ki in top_ingrs]
        response += f"Its key active ingredients include {', '.join(ingr_names)}. "

    response += f"This product is priced at {format_price(product['discount_price'])} "
    if product["discount"] > 0:
        response += f"(originally {format_price(product['normal_price'])}, {int(product['discount'] * 100)}% off) "
    response += f"and has a rating of {product['rating']} stars based on {product['review_count']} reviews."

    dataset.append({
        "instruction": instruction,
        "input": user_input,
        "output": response,
    })
    type4_count += 1

print(f"    Generated {type4_count} entries")

# --- TYPE 5: Avoidance Warnings (~70) ---
print("  Type 5: Avoidance Warnings...")
type5_count = 0

# Group ingredients by what to avoid
avoid_groups = {}
for ingr_name, info in inci_lookup.items():
    for concern in info.get("avoid", []):
        if concern not in avoid_groups:
            avoid_groups[concern] = []
        avoid_groups[concern].append({
            "name": ingr_name.title(),
            "what_does_it_do": info.get("what_does_it_do", ""),
        })

for avoid_concern, ingredients in avoid_groups.items():
    if len(ingredients) < 2:
        continue

    instruction = random.choice(AVOIDANCE_INSTRUCTIONS)

    if "Allergy" in avoid_concern:
        user_input = "I have very sensitive skin and allergies. What ingredients should I be careful with?"
        concern_display = "sensitive skin or allergies"
    elif "Pregnancy" in avoid_concern:
        user_input = "I am pregnant. What skincare ingredients should I avoid?"
        concern_display = "pregnancy"
    else:
        user_input = f"I have {avoid_concern.lower()}. What ingredients should I avoid?"
        concern_display = avoid_concern.lower()

    sample_ingrs = random.sample(ingredients, min(5, len(ingredients)))
    response = f"If you have {concern_display}, you should be cautious with the following ingredients: "
    response += ", ".join([i["name"] for i in sample_ingrs]) + ". "

    # Add a safe alternative
    safe_ingrs = [
        name.title() for name, info in inci_lookup.items()
        if avoid_concern not in info.get("avoid", []) and info.get("good_for")
    ]
    if safe_ingrs:
        safe_sample = random.sample(safe_ingrs, min(3, len(safe_ingrs)))
        response += f"Instead, look for products containing safer alternatives like {', '.join(safe_sample)}, "
        response += "which are generally well-tolerated and beneficial for the skin."

    dataset.append({
        "instruction": instruction,
        "input": user_input,
        "output": response,
    })
    type5_count += 1

# Product-specific avoidance
for product in product_profiles:
    if not product["warnings"]:
        continue

    instruction = random.choice(AVOIDANCE_INSTRUCTIONS)
    user_input = f"I have sensitive skin. Are there any concerning ingredients in the {product['brand']} {product['product_name']}?"

    warning_ingrs = []
    for ki in product["key_ingredients"]:
        if ki.get("avoid"):
            warning_ingrs.append(ki)

    response = f"The {product['brand']} {product['product_name']} contains ingredients with the following warnings: {', '.join(product['warnings'])}. "

    if warning_ingrs:
        for wi in warning_ingrs[:2]:
            response += f"{wi['name']} should be used with caution by those with {', '.join(wi['avoid']).lower()}. "

    response += "If you have sensitive skin, I recommend doing a patch test on a small area first before full application."

    dataset.append({
        "instruction": instruction,
        "input": user_input,
        "output": response,
    })
    type5_count += 1

print(f"    Generated {type5_count} entries")

# ============================================================================
# 6. SHUFFLE AND SAVE
# ============================================================================
random.shuffle(dataset)

output_path = "instruction-data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print(f"\n{'=' * 60}")
print(f"SUCCESS! Generated {len(dataset)} conversation entries.")
print(f"Saved to: {output_path}")
print(f"{'=' * 60}")

# Print breakdown
print(f"\nBreakdown:")
print(f"  Type 1 (Symptom -> Recommendation): {type1_count}")
print(f"  Type 2 (Product Inquiry):            {type2_count}")
print(f"  Type 3 (Ingredient Education):       {type3_count}")
print(f"  Type 4 (Product Claims):             {type4_count}")
print(f"  Type 5 (Avoidance Warnings):         {type5_count}")

# Preview first 3 entries
print(f"\n{'=' * 60}")
print("PREVIEW (first 3 entries):")
print(f"{'=' * 60}")
for i, entry in enumerate(dataset[:3]):
    print(f"\n--- Entry {i+1} ---")
    print(f"  Instruction: {entry['instruction']}")
    print(f"  Input: {entry['input'][:120]}...")
    print(f"  Output: {entry['output'][:200]}...")
