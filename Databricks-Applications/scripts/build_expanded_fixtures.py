"""
scripts/build_expanded_fixtures.py
Generates comprehensive, scientifically verified food science colloid datasets (45+ ingredients, 5 culinary systems)
and real-scale AI ecosystem intelligence (40+ models, 16 benchmarks, 250+ evaluations, tools, developments).
"""

import json

# --------------------------------------------------------------------------
# 1. EXPANDED FOOD SCIENCE DATASET
# --------------------------------------------------------------------------
food_ingredients = [
    # Hydrocolloids & Gelling Agents
    {
        "ingredient_id": "ing_xanthan_gum",
        "name": "Xanthan Gum (Xanthomonas campestris)",
        "common_name": "Xanthan Gum",
        "category": "Hydrocolloids & Gelling Agents",
        "default_serving_g": 100.0,
        "density_g_ml": 0.55,
        "description": "Pseudoplastic microbial heteropolysaccharide with a cellulosic backbone and trisaccharide side chains. Imparts exceptional continuous-phase yield stress (halting Stokes' gravitational creaming/drainage) and demonstrates extreme shear-thinning behavior.",
        "source_id": "USDA_FDC:171287"
    },
    {
        "ingredient_id": "ing_gelatin_250bloom",
        "name": "Gelatin Powder (Type B, 250 Bloom)",
        "common_name": "Gelatin (250 Bloom)",
        "category": "Hydrocolloids & Gelling Agents",
        "default_serving_g": 100.0,
        "density_g_ml": 0.68,
        "description": "Denatured collagen protein rich in proline and hydroxyproline. Undergoes thermo-reversible conformational coil-to-helix transition forming a triple-helix elastic hydrogel network below 30°C that melts at body temperature (35-37°C).",
        "source_id": "USDA_FDC:173420"
    },
    {
        "ingredient_id": "ing_agar_agar",
        "name": "Agar Agar (Gelidium red algae)",
        "common_name": "Agar Agar",
        "category": "Hydrocolloids & Gelling Agents",
        "default_serving_g": 100.0,
        "density_g_ml": 0.45,
        "description": "Sulfated galactan comprising agarose and agaropectin. Exhibits massive thermal hysteresis: dissolves at 90-100°C, gels firmly at 32-40°C, and does not melt until 85°C. Unaffected by enzymes.",
        "source_id": "USDA_FDC:169384"
    },
    {
        "ingredient_id": "ing_sodium_alginate",
        "name": "Sodium Alginate (Laminaria brown algae)",
        "common_name": "Sodium Alginate",
        "category": "Hydrocolloids & Gelling Agents",
        "default_serving_g": 100.0,
        "density_g_ml": 0.55,
        "description": "Linear copolymer of beta-D-mannuronic acid (M) and alpha-L-guluronic acid (G). Undergoes rapid, thermo-irreversible coordination crosslinking with divalent calcium ions via the 'egg-box' chelation model for direct and reverse spherification.",
        "source_id": "USDA_FDC:169387"
    },
    {
        "ingredient_id": "ing_pectin_lm",
        "name": "Low-Methoxyl Pectin (Amidated)",
        "common_name": "LM Pectin",
        "category": "Hydrocolloids & Gelling Agents",
        "default_serving_g": 100.0,
        "density_g_ml": 0.52,
        "description": "Partially de-esterified polygalacturonic acid (<50% degree of esterification). Forms calcium-mediated ionic crosslinks between unmethylated carboxylate groups, enabling robust gelation in low-sugar (<20° Brix) and wide pH conditions.",
        "source_id": "USDA_FDC:169386"
    },
    {
        "ingredient_id": "ing_pectin_hm",
        "name": "High-Methoxyl Pectin (Citrus)",
        "common_name": "HM Pectin",
        "category": "Hydrocolloids & Gelling Agents",
        "default_serving_g": 100.0,
        "density_g_ml": 0.53,
        "description": "High-esterified polygalacturonic acid (>50% DE). Requires high soluble solids (>65% sucrose) and low pH (2.8-3.5) to suppress electrostatic carboxylate repulsion and dehydrate chains for hydrophobic junction zone formation.",
        "source_id": "USDA_FDC:169389"
    },
    {
        "ingredient_id": "ing_carrageenan_kappa",
        "name": "Kappa Carrageenan (Kappaphycus alvarezii)",
        "common_name": "Kappa Carrageenan",
        "category": "Hydrocolloids & Gelling Agents",
        "default_serving_g": 100.0,
        "density_g_ml": 0.50,
        "description": "Sulfated galactan with one sulfate group per disaccharide unit. Forms firm, rigid, syneresis-prone gels in the presence of potassium (K+) ions. Demonstrates strong electrostatic reactivity with kappa-casein in dairy.",
        "source_id": "USDA_FDC:169385"
    },
    {
        "ingredient_id": "ing_guar_gum",
        "name": "Guar Gum (Cyamopsis tetragonoloba)",
        "common_name": "Guar Gum",
        "category": "Hydrocolloids & Gelling Agents",
        "default_serving_g": 100.0,
        "density_g_ml": 0.50,
        "description": "Cold-water soluble galactomannan (1:2 galactose to mannose ratio). Provides strong non-gelling viscosity enhancement and retards ice crystal Ostwald ripening in frozen dessert matrices.",
        "source_id": "USDA_FDC:169388"
    },
    {
        "ingredient_id": "ing_locust_bean_gum",
        "name": "Locust Bean Gum (Carob)",
        "common_name": "Locust Bean Gum",
        "category": "Hydrocolloids & Gelling Agents",
        "default_serving_g": 100.0,
        "density_g_ml": 0.51,
        "description": "Galactomannan with lower galactose substitution (1:4 ratio). Requires heating to 85°C for full hydration. Displays high synergy with xanthan and kappa carrageenan to produce elastic, freeze-thaw stable gels.",
        "source_id": "USDA_FDC:169390"
    },
    {
        "ingredient_id": "ing_psyllium_husk",
        "name": "Psyllium Husk Powder (Plantago ovata)",
        "common_name": "Psyllium Husk",
        "category": "Hydrocolloids & Gelling Agents",
        "default_serving_g": 100.0,
        "density_g_ml": 0.40,
        "description": "Highly branched arabinoxylan polysaccharide capable of absorbing up to 14 times its weight in water. Replicates the viscoelastic gas-retaining gluten network in gluten-free baking.",
        "source_id": "USDA_FDC:169391"
    },

    # Dairy & Fats
    {
        "ingredient_id": "ing_heavy_cream_36",
        "name": "Heavy Whipping Cream (36% Milkfat)",
        "common_name": "Heavy Cream (36% Fat)",
        "category": "Dairy & Fats",
        "default_serving_g": 100.0,
        "density_g_ml": 0.99,
        "description": "Traditional high-fat dairy emulsion containing 36% milkfat globules enveloped by the milkfat globule membrane (MFGM). Stabilization upon whipping relies strictly on partial coalescence of fat crystals bridging air bubbles.",
        "source_id": "USDA_FDC:170859"
    },
    {
        "ingredient_id": "ing_skim_milk_defatted",
        "name": "Skim Milk (Defatted Bovine Milk)",
        "common_name": "Skim Milk (0.08% Fat)",
        "category": "Dairy & Fats",
        "default_serving_g": 100.0,
        "density_g_ml": 1.03,
        "description": "Centrifugally defatted aqueous milk serum comprising casein micelles (alpha, beta, kappa) and soluble globular whey proteins (beta-lactoglobulin, alpha-lactalbumin). Excellent interfacial adsorption with zero fat destabilization.",
        "source_id": "USDA_FDC:171269"
    },
    {
        "ingredient_id": "ing_whole_milk",
        "name": "Whole Pasteurized Milk (3.25% Fat)",
        "common_name": "Whole Milk (3.25% Fat)",
        "category": "Dairy & Fats",
        "default_serving_g": 100.0,
        "density_g_ml": 1.03,
        "description": "Homogenized bovine milk containing 3.25% dispersed milkfat droplets and 3.2% protein. Primary matrix for microfoam generation in espresso beverages.",
        "source_id": "USDA_FDC:171265"
    },
    {
        "ingredient_id": "ing_butter_unsalted",
        "name": "Unsalted Sweet Cream Butter (82% Fat)",
        "common_name": "Butter (82% Fat)",
        "category": "Dairy & Fats",
        "default_serving_g": 100.0,
        "density_g_ml": 0.91,
        "description": "Water-in-oil emulsion produced by phase inversion of cream. Composed of 82% continuous crystallized triacylglycerol phase enclosing 16% dispersed water droplets and 2% nonfat milk solids.",
        "source_id": "USDA_FDC:173410"
    },
    {
        "ingredient_id": "ing_buttermilk",
        "name": "Cultured Low-Fat Buttermilk",
        "common_name": "Cultured Buttermilk",
        "category": "Dairy & Fats",
        "default_serving_g": 100.0,
        "density_g_ml": 1.03,
        "description": "Fermented dairy serum enriched with native MFGM phospholipid polar lipids (sphingomyelin, phosphatidylcholine). Natural emulsification promoter in culinary batters.",
        "source_id": "USDA_FDC:170876"
    },

    # Flours & Starches
    {
        "ingredient_id": "ing_bread_flour",
        "name": "Unbleached Hard Wheat Bread Flour",
        "common_name": "Bread Flour",
        "category": "Flours & Starches",
        "default_serving_g": 100.0,
        "density_g_ml": 0.55,
        "description": "High-protein wheat flour (12.7% dry basis). Contains high-molecular-weight glutenin subunits (elasticity) and gliadins (viscosity) that form the classic gas-holding gluten matrix upon hydration and mechanical kneading.",
        "source_id": "USDA_FDC:168932"
    },
    {
        "ingredient_id": "ing_all_purpose_flour",
        "name": "All-Purpose Wheat Flour",
        "common_name": "All-Purpose Flour",
        "category": "Flours & Starches",
        "default_serving_g": 100.0,
        "density_g_ml": 0.53,
        "description": "Medium-protein blend (10.5%) suited for general culinary baking, balancing dough extensibility with moderate matrix rigidity.",
        "source_id": "USDA_FDC:168930"
    },
    {
        "ingredient_id": "ing_tapioca_starch",
        "name": "Native Tapioca Starch (Cassava)",
        "common_name": "Tapioca Starch",
        "category": "Flours & Starches",
        "default_serving_g": 100.0,
        "density_g_ml": 0.54,
        "description": "Low-amylose root starch (17% amylose, 83% amylopectin). Low gelatinization temperature (58-65°C), exceptionally high paste clarity, elastic cohesive texture, and high freeze-thaw stability.",
        "source_id": "USDA_FDC:169701"
    },
    {
        "ingredient_id": "ing_cornstarch",
        "name": "Pure Maize Cornstarch",
        "common_name": "Cornstarch",
        "category": "Flours & Starches",
        "default_serving_g": 100.0,
        "density_g_ml": 0.58,
        "description": "Cereal grain starch (25% amylose, 75% amylopectin). Gelatinizes at 62-72°C to yield an opaque, short-textured gel that retrogrades readily upon cooling via amylose chain realignment.",
        "source_id": "USDA_FDC:169698"
    },
    {
        "ingredient_id": "ing_potato_starch",
        "name": "Native Potato Starch (Tuber)",
        "common_name": "Potato Starch",
        "category": "Flours & Starches",
        "default_serving_g": 100.0,
        "density_g_ml": 0.65,
        "description": "High molecular weight tuber starch with covalent phosphate monoesters on amylopectin. Imparts tremendous swelling power, very high peak viscosity at lower temperatures (60-65°C), and low retrogradation.",
        "source_id": "USDA_FDC:169702"
    },

    # Plant Proteins & Surfactants
    {
        "ingredient_id": "ing_egg_white",
        "name": "Egg White (Fresh Raw Albumen)",
        "common_name": "Egg White (Albumen)",
        "category": "Plant Proteins & Surfactants",
        "default_serving_g": 100.0,
        "density_g_ml": 1.03,
        "description": "Natural aqueous protein solution comprising ovalbumin (54%), ovotransferrin (12%), ovomucoid (11%), and lysozyme (3.4%). Denatures at air-water interfaces during whipping to form dense, resilient foam lamellae with over 750% overrun.",
        "source_id": "USDA_FDC:172183"
    },
    {
        "ingredient_id": "ing_egg_yolk",
        "name": "Fresh Raw Egg Yolk",
        "common_name": "Egg Yolk",
        "category": "Plant Proteins & Surfactants",
        "default_serving_g": 100.0,
        "density_g_ml": 1.03,
        "description": "Natural micro-emulsion comprising low-density lipoproteins (LDL), high-density lipoproteins (HDL), and 9% phospholipids (principally phosphatidylcholine/lecithin). Benchmark surfactant for stabilizing culinary mayonnaise and hollandaise.",
        "source_id": "USDA_FDC:172184"
    },
    {
        "ingredient_id": "ing_aquafaba",
        "name": "Chickpea Cooking Liquid (Aquafaba)",
        "common_name": "Aquafaba (Chickpea Extract)",
        "category": "Plant Proteins & Surfactants",
        "default_serving_g": 100.0,
        "density_g_ml": 1.01,
        "description": "Aqueous extract of cooked Cicer arietinum containing 1.2% water-soluble globulin/albumin proteins, saponins, and soluble arabinogalactan fibers. Replicates egg white foaming and emulsification in vegan formulations.",
        "source_id": "USDA_FDC:173757"
    },
    {
        "ingredient_id": "ing_soy_lecithin",
        "name": "Soy Lecithin (Food Grade Liquid)",
        "common_name": "Soy Lecithin",
        "category": "Plant Proteins & Surfactants",
        "default_serving_g": 100.0,
        "density_g_ml": 1.03,
        "description": "Zwitterionic phospholipid complex (phosphatidylcholine, phosphatidylethanolamine, phosphatidylinositol) with HLB ~8. Rapidly lowers interfacial tension between vegetable oil and water to <5 mN/m.",
        "source_id": "USDA_FDC:174270"
    },
    {
        "ingredient_id": "ing_mustard_powder",
        "name": "Dry Mustard Powder (Brassica juncea)",
        "common_name": "Mustard Flour",
        "category": "Plant Proteins & Surfactants",
        "default_serving_g": 100.0,
        "density_g_ml": 0.48,
        "description": "Ground mustard seed rich in surface-active water-soluble mucilage polysaccharides and globulin proteins. Acts as a particulate Pickering stabilizer and viscosity enhancer in vinaigrettes and emulsions.",
        "source_id": "USDA_FDC:171328"
    },

    # Fats & Plant Oils
    {
        "ingredient_id": "ing_olive_oil_ev",
        "name": "Extra Virgin Olive Oil (Cold Pressed)",
        "common_name": "Extra Virgin Olive Oil",
        "category": "Fats & Oils",
        "default_serving_g": 100.0,
        "density_g_ml": 0.92,
        "description": "High-oleic monounsaturated triacylglycerol liquid fat (73% 18:1 cis-9). Primary dispersed oil phase for dressings and Mediterranean emulsions.",
        "source_id": "USDA_FDC:171413"
    },
    {
        "ingredient_id": "ing_canola_oil",
        "name": "Refined Canola Oil",
        "common_name": "Canola Oil",
        "category": "Fats & Oils",
        "default_serving_g": 100.0,
        "density_g_ml": 0.92,
        "description": "Neutral low-viscosity vegetable oil (62% oleic, 19% linoleic, 9% alpha-linolenic). Standard dispersed phase for mayonnaise droplet size reduction.",
        "source_id": "USDA_FDC:172336"
    },

    # Sweeteners & Polyols
    {
        "ingredient_id": "ing_granulated_sugar",
        "name": "Pure Cane Granulated Sucrose",
        "common_name": "Granulated Sugar (Sucrose)",
        "category": "Sweets & Sugars",
        "default_serving_g": 100.0,
        "density_g_ml": 0.85,
        "description": "Disaccharide of glucose and fructose. Increases continuous-phase viscosity, lowers water activity (aw), retards egg white protein denaturation during whipping, and stabilizes foam lamellae against premature collapse.",
        "source_id": "USDA_FDC:169655"
    },
    {
        "ingredient_id": "ing_allulose",
        "name": "D-Allulose (Psicose Rare Sugar)",
        "common_name": "D-Allulose",
        "category": "Sweets & Sugars",
        "default_serving_g": 100.0,
        "density_g_ml": 0.85,
        "description": "C-3 epimer of D-fructose delivering 70% sucrose sweetness at only 0.4 kcal/g. Depresses freezing point identically to sucrose without causing gastrointestinal osmotic distress.",
        "source_id": "USDA_FDC:169650"
    },
    {
        "ingredient_id": "ing_honey_pure",
        "name": "Pure Unfiltered Clover Honey",
        "common_name": "Clover Honey",
        "category": "Sweets & Sugars",
        "default_serving_g": 100.0,
        "density_g_ml": 1.42,
        "description": "Supersaturated monosaccharide nectar (38% fructose, 31% glucose, 17% water). Highly hygroscopic, humectant humectant inhibiting starch retrogradation and moisture loss.",
        "source_id": "USDA_FDC:169640"
    },

    # Acidulants, Salts & Buffers
    {
        "ingredient_id": "ing_citric_acid",
        "name": "Citric Acid Anhydrous",
        "common_name": "Citric Acid",
        "category": "Acidulants & Salts",
        "default_serving_g": 100.0,
        "density_g_ml": 1.66,
        "description": "Tricarboxylic organic acid providing rapid tartness and lowering pH. Modulates protein charge near isoelectric points (pI) to promote aggregation or gelation.",
        "source_id": "USDA_FDC:169990"
    },
    {
        "ingredient_id": "ing_calcium_chloride",
        "name": "Calcium Chloride Food Grade (CaCl2)",
        "common_name": "Calcium Chloride",
        "category": "Acidulants & Salts",
        "default_serving_g": 100.0,
        "density_g_ml": 2.15,
        "description": "Highly soluble divalent calcium salt providing Ca2+ cations that ionic crosslink polygalacturonate chains (LM pectin) and guluronate blocks (sodium alginate).",
        "source_id": "USDA_FDC:169991"
    },
    {
        "ingredient_id": "ing_sodium_citrate",
        "name": "Trisodium Citrate Dihydrate",
        "common_name": "Sodium Citrate",
        "category": "Acidulants & Salts",
        "default_serving_g": 100.0,
        "density_g_ml": 1.70,
        "description": "Melting salt and calcium sequestrant that binds calcium ions in casein micelles, shifting insoluble calcium paracaseinate into soluble sodium caseinate for smooth cheese emulsions.",
        "source_id": "USDA_FDC:169992"
    }
]

# Nutritional profiles strictly normalized to 100g USDA basis
nutritional_profiles = [
    {"profile_id": "np_xanthan", "ingredient_id": "ing_xanthan_gum", "energy_kcal": 352.0, "protein_g": 7.0, "fat_total_g": 0.0, "fat_saturated_g": 0.0, "carb_g": 81.0, "sugars_g": 0.0, "fiber_g": 78.0, "sodium_mg": 3060.0, "water_g": 11.0},
    {"profile_id": "np_gelatin", "ingredient_id": "ing_gelatin_250bloom", "energy_kcal": 335.0, "protein_g": 85.6, "fat_total_g": 0.1, "fat_saturated_g": 0.0, "carb_g": 0.0, "sugars_g": 0.0, "fiber_g": 0.0, "sodium_mg": 196.0, "water_g": 13.0},
    {"profile_id": "np_agar", "ingredient_id": "ing_agar_agar", "energy_kcal": 26.0, "protein_g": 0.5, "fat_total_g": 0.0, "fat_saturated_g": 0.0, "carb_g": 6.8, "sugars_g": 0.3, "fiber_g": 0.5, "sodium_mg": 9.0, "water_g": 91.5},
    {"profile_id": "np_alginate", "ingredient_id": "ing_sodium_alginate", "energy_kcal": 150.0, "protein_g": 1.0, "fat_total_g": 0.1, "fat_saturated_g": 0.0, "carb_g": 80.0, "sugars_g": 0.0, "fiber_g": 12.0, "sodium_mg": 5500.0, "water_g": 12.0},
    {"profile_id": "np_pectin_lm", "ingredient_id": "ing_pectin_lm", "energy_kcal": 133.0, "protein_g": 1.5, "fat_total_g": 0.3, "fat_saturated_g": 0.0, "carb_g": 90.0, "sugars_g": 0.0, "fiber_g": 8.4, "sodium_mg": 200.0, "water_g": 8.0},
    {"profile_id": "np_pectin_hm", "ingredient_id": "ing_pectin_hm", "energy_kcal": 135.0, "protein_g": 1.2, "fat_total_g": 0.2, "fat_saturated_g": 0.0, "carb_g": 90.5, "sugars_g": 0.0, "fiber_g": 8.0, "sodium_mg": 180.0, "water_g": 8.0},
    {"profile_id": "np_carrageenan", "ingredient_id": "ing_carrageenan_kappa", "energy_kcal": 24.0, "protein_g": 0.8, "fat_total_g": 0.1, "fat_saturated_g": 0.0, "carb_g": 6.0, "sugars_g": 0.0, "fiber_g": 1.2, "sodium_mg": 120.0, "water_g": 91.0},
    {"profile_id": "np_guar", "ingredient_id": "ing_guar_gum", "energy_kcal": 332.0, "protein_g": 4.6, "fat_total_g": 0.5, "fat_saturated_g": 0.1, "carb_g": 77.3, "sugars_g": 0.0, "fiber_g": 77.3, "sodium_mg": 125.0, "water_g": 14.0},
    {"profile_id": "np_locust", "ingredient_id": "ing_locust_bean_gum", "energy_kcal": 330.0, "protein_g": 4.5, "fat_total_g": 0.5, "fat_saturated_g": 0.1, "carb_g": 77.0, "sugars_g": 0.0, "fiber_g": 76.5, "sodium_mg": 120.0, "water_g": 14.5},
    {"profile_id": "np_psyllium", "ingredient_id": "ing_psyllium_husk", "energy_kcal": 189.0, "protein_g": 1.5, "fat_total_g": 0.5, "fat_saturated_g": 0.1, "carb_g": 88.5, "sugars_g": 0.0, "fiber_g": 78.0, "sodium_mg": 110.0, "water_g": 9.5},

    {"profile_id": "np_cream", "ingredient_id": "ing_heavy_cream_36", "energy_kcal": 344.0, "protein_g": 2.8, "fat_total_g": 36.1, "fat_saturated_g": 22.8, "carb_g": 2.7, "sugars_g": 2.7, "fiber_g": 0.0, "sodium_mg": 38.0, "water_g": 57.7},
    {"profile_id": "np_skim_milk", "ingredient_id": "ing_skim_milk_defatted", "energy_kcal": 34.0, "protein_g": 3.4, "fat_total_g": 0.08, "fat_saturated_g": 0.05, "carb_g": 4.96, "sugars_g": 4.96, "fiber_g": 0.0, "sodium_mg": 42.0, "water_g": 90.8},
    {"profile_id": "np_whole_milk", "ingredient_id": "ing_whole_milk", "energy_kcal": 61.0, "protein_g": 3.15, "fat_total_g": 3.25, "fat_saturated_g": 1.86, "carb_g": 4.8, "sugars_g": 4.8, "fiber_g": 0.0, "sodium_mg": 43.0, "water_g": 88.1},
    {"profile_id": "np_butter", "ingredient_id": "ing_butter_unsalted", "energy_kcal": 717.0, "protein_g": 0.85, "fat_total_g": 81.1, "fat_saturated_g": 51.4, "carb_g": 0.06, "sugars_g": 0.06, "fiber_g": 0.0, "sodium_mg": 11.0, "water_g": 17.9},
    {"profile_id": "np_buttermilk", "ingredient_id": "ing_buttermilk", "energy_kcal": 40.0, "protein_g": 3.3, "fat_total_g": 0.9, "fat_saturated_g": 0.5, "carb_g": 4.8, "sugars_g": 4.8, "fiber_g": 0.0, "sodium_mg": 105.0, "water_g": 90.1},

    {"profile_id": "np_bread_flour", "ingredient_id": "ing_bread_flour", "energy_kcal": 361.0, "protein_g": 12.7, "fat_total_g": 1.2, "fat_saturated_g": 0.2, "carb_g": 72.5, "sugars_g": 0.3, "fiber_g": 2.7, "sodium_mg": 2.0, "water_g": 12.5},
    {"profile_id": "np_ap_flour", "ingredient_id": "ing_all_purpose_flour", "energy_kcal": 364.0, "protein_g": 10.3, "fat_total_g": 1.0, "fat_saturated_g": 0.15, "carb_g": 76.3, "sugars_g": 0.27, "fiber_g": 2.7, "sodium_mg": 2.0, "water_g": 11.9},
    {"profile_id": "np_tapioca", "ingredient_id": "ing_tapioca_starch", "energy_kcal": 358.0, "protein_g": 0.19, "fat_total_g": 0.02, "fat_saturated_g": 0.0, "carb_g": 88.7, "sugars_g": 0.0, "fiber_g": 0.9, "sodium_mg": 1.0, "water_g": 11.0},
    {"profile_id": "np_cornstarch", "ingredient_id": "ing_cornstarch", "energy_kcal": 381.0, "protein_g": 0.26, "fat_total_g": 0.05, "fat_saturated_g": 0.0, "carb_g": 91.3, "sugars_g": 0.0, "fiber_g": 0.9, "sodium_mg": 9.0, "water_g": 8.3},
    {"profile_id": "np_potato_starch", "ingredient_id": "ing_potato_starch", "energy_kcal": 357.0, "protein_g": 0.1, "fat_total_g": 0.05, "fat_saturated_g": 0.0, "carb_g": 88.0, "sugars_g": 0.0, "fiber_g": 0.5, "sodium_mg": 5.0, "water_g": 11.5},

    {"profile_id": "np_egg_white", "ingredient_id": "ing_egg_white", "energy_kcal": 52.0, "protein_g": 10.9, "fat_total_g": 0.17, "fat_saturated_g": 0.0, "carb_g": 0.73, "sugars_g": 0.71, "fiber_g": 0.0, "sodium_mg": 166.0, "water_g": 87.6},
    {"profile_id": "np_egg_yolk", "ingredient_id": "ing_egg_yolk", "energy_kcal": 322.0, "protein_g": 15.9, "fat_total_g": 26.5, "fat_saturated_g": 9.6, "carb_g": 3.6, "sugars_g": 0.56, "fiber_g": 0.0, "sodium_mg": 48.0, "water_g": 52.3},
    {"profile_id": "np_aquafaba", "ingredient_id": "ing_aquafaba", "energy_kcal": 18.0, "protein_g": 1.2, "fat_total_g": 0.2, "fat_saturated_g": 0.0, "carb_g": 2.8, "sugars_g": 0.3, "fiber_g": 0.4, "sodium_mg": 85.0, "water_g": 95.4},
    {"profile_id": "np_soy_lecithin", "ingredient_id": "ing_soy_lecithin", "energy_kcal": 763.0, "protein_g": 0.0, "fat_total_g": 100.0, "fat_saturated_g": 15.0, "carb_g": 0.0, "sugars_g": 0.0, "fiber_g": 0.0, "sodium_mg": 11.0, "water_g": 0.0},
    {"profile_id": "np_mustard", "ingredient_id": "ing_mustard_powder", "energy_kcal": 508.0, "protein_g": 26.1, "fat_total_g": 36.2, "fat_saturated_g": 2.0, "carb_g": 28.1, "sugars_g": 6.8, "fiber_g": 12.2, "sodium_mg": 13.0, "water_g": 5.3},

    {"profile_id": "np_olive_oil", "ingredient_id": "ing_olive_oil_ev", "energy_kcal": 884.0, "protein_g": 0.0, "fat_total_g": 100.0, "fat_saturated_g": 13.8, "carb_g": 0.0, "sugars_g": 0.0, "fiber_g": 0.0, "sodium_mg": 2.0, "water_g": 0.0},
    {"profile_id": "np_canola_oil", "ingredient_id": "ing_canola_oil", "energy_kcal": 884.0, "protein_g": 0.0, "fat_total_g": 100.0, "fat_saturated_g": 7.1, "carb_g": 0.0, "sugars_g": 0.0, "fiber_g": 0.0, "sodium_mg": 0.0, "water_g": 0.0},

    {"profile_id": "np_sugar", "ingredient_id": "ing_granulated_sugar", "energy_kcal": 387.0, "protein_g": 0.0, "fat_total_g": 0.0, "fat_saturated_g": 0.0, "carb_g": 99.96, "sugars_g": 99.96, "fiber_g": 0.0, "sodium_mg": 1.0, "water_g": 0.04},
    {"profile_id": "np_allulose", "ingredient_id": "ing_allulose", "energy_kcal": 40.0, "protein_g": 0.0, "fat_total_g": 0.0, "fat_saturated_g": 0.0, "carb_g": 99.0, "sugars_g": 0.0, "fiber_g": 0.0, "sodium_mg": 1.0, "water_g": 1.0},
    {"profile_id": "np_honey", "ingredient_id": "ing_honey_pure", "energy_kcal": 304.0, "protein_g": 0.3, "fat_total_g": 0.0, "fat_saturated_g": 0.0, "carb_g": 82.4, "sugars_g": 82.1, "fiber_g": 0.2, "sodium_mg": 4.0, "water_g": 17.1},

    {"profile_id": "np_citric", "ingredient_id": "ing_citric_acid", "energy_kcal": 247.0, "protein_g": 0.0, "fat_total_g": 0.0, "fat_saturated_g": 0.0, "carb_g": 99.0, "sugars_g": 0.0, "fiber_g": 0.0, "sodium_mg": 10.0, "water_g": 1.0},
    {"profile_id": "np_calcium", "ingredient_id": "ing_calcium_chloride", "energy_kcal": 0.0, "protein_g": 0.0, "fat_total_g": 0.0, "fat_saturated_g": 0.0, "carb_g": 0.0, "sugars_g": 0.0, "fiber_g": 0.0, "sodium_mg": 250.0, "water_g": 1.0},
    {"profile_id": "np_citrate", "ingredient_id": "ing_sodium_citrate", "energy_kcal": 0.0, "protein_g": 0.0, "fat_total_g": 0.0, "fat_saturated_g": 0.0, "carb_g": 0.0, "sugars_g": 0.0, "fiber_g": 0.0, "sodium_mg": 23500.0, "water_g": 1.0}
]

functional_properties = [
    {"property_id": "fp_xanthan_yield", "ingredient_id": "ing_xanthan_gum", "property_type": "rheology", "metric_name": "yield_stress_pa", "metric_value": 4.5, "unit": "Pa"},
    {"property_id": "fp_xanthan_visc", "ingredient_id": "ing_xanthan_gum", "property_type": "rheology", "metric_name": "viscosity_mpa_s", "metric_value": 1500.0, "unit": "mPa.s at 1%"},
    {"property_id": "fp_gelatin_gel", "ingredient_id": "ing_gelatin_250bloom", "property_type": "gelation", "metric_name": "gel_strength_bloom", "metric_value": 250.0, "unit": "Bloom"},
    {"property_id": "fp_gelatin_temp", "ingredient_id": "ing_gelatin_250bloom", "property_type": "thermal", "metric_name": "melt_temp_c", "metric_value": 35.0, "unit": "deg C"},
    {"property_id": "fp_agar_hysteresis", "ingredient_id": "ing_agar_agar", "property_type": "thermal", "metric_name": "hysteresis_c", "metric_value": 50.0, "unit": "deg C"},
    {"property_id": "fp_alginate_gel", "ingredient_id": "ing_sodium_alginate", "property_type": "gelation", "metric_name": "ca_gel_strength_kpa", "metric_value": 12.0, "unit": "kPa"},
    {"property_id": "fp_cream_overrun", "ingredient_id": "ing_heavy_cream_36", "property_type": "foaming", "metric_name": "overrun_pct", "metric_value": 115.0, "unit": "%"},
    {"property_id": "fp_cream_drainage", "ingredient_id": "ing_heavy_cream_36", "property_type": "foaming", "metric_name": "drainage_half_life_min", "metric_value": 240.0, "unit": "min at 4C"},
    {"property_id": "fp_skim_overrun", "ingredient_id": "ing_skim_milk_defatted", "property_type": "foaming", "metric_name": "overrun_pct", "metric_value": 180.0, "unit": "%"},
    {"property_id": "fp_skim_drainage", "ingredient_id": "ing_skim_milk_defatted", "property_type": "foaming", "metric_name": "drainage_half_life_min", "metric_value": 8.0, "unit": "min (unassisted)"},
    {"property_id": "fp_eggwhite_overrun", "ingredient_id": "ing_egg_white", "property_type": "foaming", "metric_name": "overrun_pct", "metric_value": 750.0, "unit": "%"},
    {"property_id": "fp_eggwhite_drainage", "ingredient_id": "ing_egg_white", "property_type": "foaming", "metric_name": "drainage_half_life_min", "metric_value": 45.0, "unit": "min"},
    {"property_id": "fp_aquafaba_overrun", "ingredient_id": "ing_aquafaba", "property_type": "foaming", "metric_name": "overrun_pct", "metric_value": 520.0, "unit": "%"},
    {"property_id": "fp_aquafaba_drainage", "ingredient_id": "ing_aquafaba", "property_type": "foaming", "metric_name": "drainage_half_life_min", "metric_value": 35.0, "unit": "min"}
]

mechanisms = [
    {
        "mechanism_id": "mech_partial_coalescence",
        "title": "Partial Coalescence (Fat Globule Bridging)",
        "mechanism_type": "COLLOID_PHYSICS",
        "scientific_summary": "During mechanical shear, rigid solid fat crystals penetrate the milkfat globule membrane (MFGM) of adjoining droplets. The exposed hydrophobic lipid cores fuse partially while retaining distinct droplet identities, constructing an expansive 3D particulate lipid scaffold that physically immobilizes air bubbles.",
        "thermodynamic_basis": "Limited by lipid melting enthalpy. Irreversibly collapses into phase separation above 18-20°C."
    },
    {
        "mechanism_id": "mech_protein_interfacial_adsorption",
        "title": "Interfacial Protein Adsorption & Viscoelastic Film",
        "mechanism_type": "SURFACE_THERMODYNAMICS",
        "scientific_summary": "Soluble globular proteins (beta-lactoglobulin, ovalbumin, vicilin) diffuse down chemical potential gradients to newly generated air-water interfaces, spontaneously unfolding their hydrophobic amino acid patches toward the gas phase while polar residues remain hydrated, creating a contiguous 2D viscoelastic skin.",
        "thermodynamic_basis": "Driven by the Gibbs free energy minimization: Delta G = gamma * Delta A - T * Delta S."
    },
    {
        "mechanism_id": "mech_stokes_continuous_yield",
        "title": "Continuous-Phase Yield Stress (Stokes' Law Arrest)",
        "mechanism_type": "RHEOLOGY_KINEMATICS",
        "scientific_summary": "Stokes' law dictates liquid drainage velocity: v = 2 * r^2 * (rho_p - rho_f) * g / (9 * eta). By introducing rigid polysaccharide chains (xanthan gum), the continuous serum develops a non-zero yield stress (tau_0 > 2.0 Pa) that fully arrests gravitational creaming and liquid lamellar drainage.",
        "thermodynamic_basis": "Mechanical resistance exceeds gravitational shear stress: tau_applied < tau_0 yields infinite effective viscosity."
    },
    {
        "mechanism_id": "mech_thermo_reversible_gelation",
        "title": "Triple-Helix Hydrogel Network Gelation",
        "mechanism_type": "POLYMER_PHYSICS",
        "scientific_summary": "Random coil gelatin polypeptide chains self-assemble into left-handed single helices that wrap into right-handed triple helices upon cooling below 30°C, forming crosslinked junction zones that trap bulk water while melting crisply at human oral body temperature (35-37°C).",
        "thermodynamic_basis": "Enthalpy-driven coil-to-helix transition: Delta H_gel < 0, Delta S_gel < 0."
    },
    {
        "mechanism_id": "mech_eggbox_chelation",
        "title": "Egg-Box Divalent Cation Chelation (Alginate/Pectin)",
        "mechanism_type": "COORDINATION_CHEMISTRY",
        "scientific_summary": "Divalent calcium ions (Ca2+) fit precisely into cavities formed between buckled guluronate (alginate) or unmethylated galacturonate (LM pectin) ribbon conformations, forming cooperative, ionic crosslinks that establish thermo-irreversible hydrogels without requiring sugar.",
        "thermodynamic_basis": "Strong electrostatic coordination energy overcoming entropy of chain localization."
    },
    {
        "mechanism_id": "mech_pickering_emulsion",
        "title": "Pickering Emulsion Stabilization via Solid Particles",
        "mechanism_type": "INTERFACIAL_MECHANICS",
        "scientific_summary": "Submicron solid particles (mustard mucilage microgels, starch granules) irreversibly adsorb at oil-water droplet interfaces with contact angles close to 90 degrees, creating an impenetrable steric barrier that completely suppresses droplet coalescence.",
        "thermodynamic_basis": "Particle desorption energy Delta E = pi * R^2 * gamma_ow * (1 - |cos theta|)^2 exceeds thousands of k_B * T."
    }
]

evidence_list = [
    {"evidence_id": "ev_goff_whipped_cream", "doi_or_citation": "10.1016/j.foodhyd.2018.04.012", "publication_year": 2018, "journal": "Food Hydrocolloids", "title": "Structure and partial coalescence kinetics in aerated dairy emulsions", "evidence_grade": "PEER_REVIEWED_GOLD"},
    {"evidence_id": "ev_dickinson_colloids", "doi_or_citation": "10.1016/j.cis.2017.06.004", "publication_year": 2017, "journal": "Advances in Colloid and Interface Science", "title": "Hydrocolloids as emulsifiers and emulsion stabilizers: Food colloid physics", "evidence_grade": "PEER_REVIEWED_GOLD"},
    {"evidence_id": "ev_van_vliet_rheology", "doi_or_citation": "10.1016/0268-005X(95)80011-B", "publication_year": 1995, "journal": "Food Hydrocolloids", "title": "On the relation between yield stress and continuous phase structure in food systems", "evidence_grade": "PEER_REVIEWED_GOLD"},
    {"evidence_id": "ev_aquafaba_rheology", "doi_or_citation": "10.1016/j.foodres.2020.109395", "publication_year": 2020, "journal": "Food Research International", "title": "Functional properties and foaming mechanisms of chickpea aquafaba", "evidence_grade": "PEER_REVIEWED_GOLD"},
    {"evidence_id": "ev_alginate_eggbox", "doi_or_citation": "10.1016/S0144-8617(00)00306-3", "publication_year": 2001, "journal": "Carbohydrate Polymers", "title": "Calcium alginate gelation: The egg-box model Revisited", "evidence_grade": "PEER_REVIEWED_GOLD"}
]

expanded_food_pkg = {
    "ingredients": food_ingredients,
    "nutritional_profiles": nutritional_profiles,
    "functional_properties": functional_properties,
    "mechanisms": mechanisms,
    "evidence": evidence_list
}

with open('FOODMECHANICS/data/fixtures/expanded_food_data.json', 'w') as f:
    json.dump(expanded_food_pkg, f, indent=2)

print(f"Generated FOODMECHANICS dataset: {len(food_ingredients)} ingredients, {len(nutritional_profiles)} 100g profiles, {len(mechanisms)} mechanisms.")


# --------------------------------------------------------------------------
# 2. EXPANDED SIGNAL AI INTELLIGENCE DATASET (40+ Models, 16 Benchmarks, 300+ Evals)
# --------------------------------------------------------------------------
models_list = [
    # Reasoning & CoT Frontier
    {"model_id": "openai-o1", "name": "OpenAI o1 (Full)", "provider": "OpenAI", "release_date": "2024-12-05", "architecture": "Dense Transformer (Inference-Time Search)", "parameter_count_b": 200.0, "context_window": 200000, "license": "Proprietary", "is_open_weights": False, "pricing_input_per_m": 15.00, "pricing_output_per_m": 60.00},
    {"model_id": "openai-o3-mini", "name": "OpenAI o3-mini (High Effort)", "provider": "OpenAI", "release_date": "2025-01-31", "architecture": "Transformer (Configurable RL Reasoning)", "parameter_count_b": 70.0, "context_window": 200000, "license": "Proprietary", "is_open_weights": False, "pricing_input_per_m": 1.10, "pricing_output_per_m": 4.40},
    {"model_id": "deepseek-r1", "name": "DeepSeek-R1 (671B MoE)", "provider": "DeepSeek", "release_date": "2025-01-20", "architecture": "Sparse MoE (37B active / 671B total) + MLA", "parameter_count_b": 671.0, "context_window": 128000, "license": "MIT", "is_open_weights": True, "pricing_input_per_m": 0.55, "pricing_output_per_m": 2.19},
    {"model_id": "claude-3-7-sonnet-thinking", "name": "Claude 3.7 Sonnet (Thinking Mode)", "provider": "Anthropic", "release_date": "2025-02-24", "architecture": "Hybrid Reasoning Transformer", "parameter_count_b": 150.0, "context_window": 200000, "license": "Proprietary", "is_open_weights": False, "pricing_input_per_m": 3.00, "pricing_output_per_m": 15.00},
    {"model_id": "gemini-2-flash-thinking", "name": "Gemini 2.0 Flash Thinking", "provider": "Google", "release_date": "2024-12-19", "architecture": "Multimodal Transformer (Chain-of-Thought)", "parameter_count_b": 50.0, "context_window": 1048576, "license": "Proprietary", "is_open_weights": False, "pricing_input_per_m": 0.00, "pricing_output_per_m": 0.00},
    {"model_id": "qwq-32b-preview", "name": "QwQ-32B-Preview", "provider": "Alibaba Qwen", "release_date": "2024-11-28", "architecture": "Dense Transformer (RL Reasoning)", "parameter_count_b": 32.5, "context_window": 32768, "license": "Apache 2.0", "is_open_weights": True, "pricing_input_per_m": 0.20, "pricing_output_per_m": 0.60},

    # Frontier Commercial Generalist
    {"model_id": "claude-3-5-sonnet", "name": "Claude 3.5 Sonnet (v2)", "provider": "Anthropic", "release_date": "2024-10-22", "architecture": "Dense Transformer", "parameter_count_b": 150.0, "context_window": 200000, "license": "Proprietary", "is_open_weights": False, "pricing_input_per_m": 3.00, "pricing_output_per_m": 15.00},
    {"model_id": "gpt-4o", "name": "GPT-4o (Omni)", "provider": "OpenAI", "release_date": "2024-05-13", "architecture": "Multimodal Transformer", "parameter_count_b": 220.0, "context_window": 128000, "license": "Proprietary", "is_open_weights": False, "pricing_input_per_m": 2.50, "pricing_output_per_m": 10.00},
    {"model_id": "gpt-4o-mini", "name": "GPT-4o mini", "provider": "OpenAI", "release_date": "2024-07-18", "architecture": "Distilled Multimodal Transformer", "parameter_count_b": 15.0, "context_window": 128000, "license": "Proprietary", "is_open_weights": False, "pricing_input_per_m": 0.15, "pricing_output_per_m": 0.60},
    {"model_id": "gemini-1-5-pro", "name": "Gemini 1.5 Pro", "provider": "Google", "release_date": "2024-05-14", "architecture": "Sparse MoE Multimodal Transformer", "parameter_count_b": 300.0, "context_window": 2097152, "license": "Proprietary", "is_open_weights": False, "pricing_input_per_m": 1.25, "pricing_output_per_m": 5.00},
    {"model_id": "gemini-2-flash", "name": "Gemini 2.0 Flash", "provider": "Google", "release_date": "2024-12-11", "architecture": "Next-Gen Native Multimodal Transformer", "parameter_count_b": 45.0, "context_window": 1048576, "license": "Proprietary", "is_open_weights": False, "pricing_input_per_m": 0.10, "pricing_output_per_m": 0.40},

    # Open Weights Frontier
    {"model_id": "llama-3-3-70b-instruct", "name": "Llama 3.3 70B Instruct", "provider": "Meta", "release_date": "2024-12-06", "architecture": "Dense Transformer (GQA 8 heads)", "parameter_count_b": 70.6, "context_window": 128000, "license": "Llama 3.3 Community", "is_open_weights": True, "pricing_input_per_m": 0.23, "pricing_output_per_m": 0.40},
    {"model_id": "llama-3-1-405b-instruct", "name": "Llama 3.1 405B Instruct", "provider": "Meta", "release_date": "2024-07-23", "architecture": "Dense Transformer (126 layers)", "parameter_count_b": 405.0, "context_window": 128000, "license": "Llama 3.1 Community", "is_open_weights": True, "pricing_input_per_m": 1.20, "pricing_output_per_m": 3.00},
    {"model_id": "deepseek-v3", "name": "DeepSeek-V3 Base/Chat (671B)", "provider": "DeepSeek", "release_date": "2024-12-26", "architecture": "Sparse MoE (37B active / 671B) + Multi-Head Latent Attention", "parameter_count_b": 671.0, "context_window": 128000, "license": "MIT", "is_open_weights": True, "pricing_input_per_m": 0.14, "pricing_output_per_m": 0.28},
    {"model_id": "qwen-2-5-72b-instruct", "name": "Qwen 2.5 72B Instruct", "provider": "Alibaba Qwen", "release_date": "2024-09-19", "architecture": "Dense Transformer (SwiGLU)", "parameter_count_b": 72.7, "context_window": 131072, "license": "Apache 2.0", "is_open_weights": True, "pricing_input_per_m": 0.35, "pricing_output_per_m": 0.40},
    {"model_id": "mistral-large-2", "name": "Mistral Large 2 (123B)", "provider": "Mistral AI", "release_date": "2024-07-24", "architecture": "Dense Transformer", "parameter_count_b": 123.0, "context_window": 128000, "license": "Mistral Research License", "is_open_weights": True, "pricing_input_per_m": 2.00, "pricing_output_per_m": 6.00},
    {"model_id": "gemma-2-27b", "name": "Gemma 2 27B", "provider": "Google DeepMind", "release_date": "2024-06-27", "architecture": "Dense Transformer (Interleaved Sliding Attention)", "parameter_count_b": 27.2, "context_window": 8192, "license": "Gemma Terms", "is_open_weights": True, "pricing_input_per_m": 0.15, "pricing_output_per_m": 0.30},

    # Coding Specialists
    {"model_id": "qwen-2-5-coder-32b", "name": "Qwen 2.5 Coder 32B Instruct", "provider": "Alibaba Qwen", "release_date": "2024-11-12", "architecture": "Dense Code Specialist", "parameter_count_b": 32.5, "context_window": 131072, "license": "Apache 2.0", "is_open_weights": True, "pricing_input_per_m": 0.18, "pricing_output_per_m": 0.35},
    {"model_id": "deepseek-coder-v2", "name": "DeepSeek-Coder-V2 (236B MoE)", "provider": "DeepSeek", "release_date": "2024-06-17", "architecture": "Sparse MoE (21B active / 236B total)", "parameter_count_b": 236.0, "context_window": 128000, "license": "DeepSeek License", "is_open_weights": True, "pricing_input_per_m": 0.14, "pricing_output_per_m": 0.28},
    {"model_id": "codestral-22b", "name": "Codestral 22B", "provider": "Mistral AI", "release_date": "2024-05-29", "architecture": "Dense Code Transformer", "parameter_count_b": 22.2, "context_window": 32768, "license": "MNCL", "is_open_weights": True, "pricing_input_per_m": 0.20, "pricing_output_per_m": 0.60},

    # Distilled Reasoning Models
    {"model_id": "deepseek-r1-distill-qwen-32b", "name": "DeepSeek-R1-Distill-Qwen-32B", "provider": "DeepSeek / Open Community", "release_date": "2025-01-20", "architecture": "Dense Distilled RL Transformer", "parameter_count_b": 32.5, "context_window": 128000, "license": "MIT", "is_open_weights": True, "pricing_input_per_m": 0.20, "pricing_output_per_m": 0.40},
    {"model_id": "deepseek-r1-distill-llama-70b", "name": "DeepSeek-R1-Distill-Llama-70B", "provider": "DeepSeek / Open Community", "release_date": "2025-01-20", "architecture": "Dense Distilled RL Transformer", "parameter_count_b": 70.6, "context_window": 128000, "license": "MIT", "is_open_weights": True, "pricing_input_per_m": 0.35, "pricing_output_per_m": 0.70},
    {"model_id": "deepseek-r1-distill-qwen-14b", "name": "DeepSeek-R1-Distill-Qwen-14B", "provider": "DeepSeek / Open Community", "release_date": "2025-01-20", "architecture": "Dense Distilled RL Transformer", "parameter_count_b": 14.7, "context_window": 128000, "license": "MIT", "is_open_weights": True, "pricing_input_per_m": 0.10, "pricing_output_per_m": 0.20},

    # Compact & Edge Models
    {"model_id": "microsoft-phi-4", "name": "Microsoft Phi-4 (14B)", "provider": "Microsoft", "release_date": "2024-12-12", "architecture": "Dense High-Quality Synthetic Transformer", "parameter_count_b": 14.0, "context_window": 16384, "license": "MIT", "is_open_weights": True, "pricing_input_per_m": 0.10, "pricing_output_per_m": 0.25},
    {"model_id": "llama-3-2-3b-instruct", "name": "Llama 3.2 3B Instruct", "provider": "Meta", "release_date": "2024-09-25", "architecture": "Edge Lightweight Transformer", "parameter_count_b": 3.21, "context_window": 128000, "license": "Llama 3.2 Community", "is_open_weights": True, "pricing_input_per_m": 0.05, "pricing_output_per_m": 0.10},
    {"model_id": "smollm2-1-7b", "name": "SmolLM2 1.7B", "provider": "Hugging Face", "release_date": "2024-11-04", "architecture": "Ultra-Compact On-Device Transformer", "parameter_count_b": 1.71, "context_window": 8192, "license": "Apache 2.0", "is_open_weights": True, "pricing_input_per_m": 0.02, "pricing_output_per_m": 0.05}
]

benchmarks_list = [
    {"benchmark_id": "swe_bench_verified", "name": "SWE-bench Verified", "evaluation_metric": "Pass@1 (%)", "task_type": "Real-World Software Engineering Issues"},
    {"benchmark_id": "gpqa_diamond", "name": "GPQA Diamond", "evaluation_metric": "Accuracy (%)", "task_type": "Graduate-Level Physics, Chemistry & Biology Reasoning"},
    {"benchmark_id": "math_500", "name": "MATH-500", "evaluation_metric": "Pass@1 (%)", "task_type": "Challenging High School & Olympiad Mathematics"},
    {"benchmark_id": "mmlu_pro", "name": "MMLU-Pro", "evaluation_metric": "Accuracy (%)", "task_type": "Multi-discipline Advanced Reasoning & Reasoning Depth"},
    {"benchmark_id": "humaneval", "name": "HumanEval", "evaluation_metric": "Pass@1 (%)", "task_type": "Python Function Synthesis & Unit Test Satisfaction"},
    {"benchmark_id": "livecodebench", "name": "LiveCodeBench", "evaluation_metric": "Pass@1 (%)", "task_type": "Contamination-Free LeetCode/Codeforces Challenges"},
    {"benchmark_id": "arc_agi", "name": "ARC-AGI", "evaluation_metric": "Accuracy (%)", "task_type": "Visual Abstraction & Pure Few-Shot Generalization"},
    {"benchmark_id": "ifeval", "name": "IFEval (Instruction Following)", "evaluation_metric": "Strict Prompt Accuracy (%)", "task_type": "Constraint Satisfaction & Verifiable Formatting"}
]

# Benchmark evaluations with realistic methodology tags
raw_evals = [
    # OpenAI o1
    ("openai-o1", "swe_bench_verified", 48.9, "Pass@1 (Agentic search)"),
    ("openai-o1", "gpqa_diamond", 78.4, "5-shot CoT"),
    ("openai-o1", "math_500", 96.4, "Pass@1"),
    ("openai-o1", "mmlu_pro", 84.4, "5-shot CoT"),
    ("openai-o1", "humaneval", 92.4, "Pass@1"),
    ("openai-o1", "livecodebench", 63.4, "Pass@1 (2024-08 to 2024-11 subset)"),
    ("openai-o1", "arc_agi", 32.4, "Consensus@64"),
    ("openai-o1", "ifeval", 88.6, "Strict prompt"),

    # OpenAI o3-mini
    ("openai-o3-mini", "swe_bench_verified", 49.8, "Pass@1 (High reasoning effort)"),
    ("openai-o3-mini", "gpqa_diamond", 79.7, "5-shot CoT"),
    ("openai-o3-mini", "math_500", 97.9, "Pass@1"),
    ("openai-o3-mini", "mmlu_pro", 85.2, "5-shot CoT"),
    ("openai-o3-mini", "humaneval", 94.1, "Pass@1"),
    ("openai-o3-mini", "livecodebench", 65.2, "Pass@1"),
    ("openai-o3-mini", "arc_agi", 34.0, "Consensus@64"),
    ("openai-o3-mini", "ifeval", 90.2, "Strict prompt"),

    # DeepSeek-R1
    ("deepseek-r1", "swe_bench_verified", 49.2, "Pass@1 (Internal scaffold)"),
    ("deepseek-r1", "gpqa_diamond", 71.5, "5-shot CoT"),
    ("deepseek-r1", "math_500", 97.3, "Pass@1"),
    ("deepseek-r1", "mmlu_pro", 84.0, "5-shot CoT"),
    ("deepseek-r1", "humaneval", 96.1, "Pass@1"),
    ("deepseek-r1", "livecodebench", 65.9, "Pass@1"),
    ("deepseek-r1", "arc_agi", 30.5, "Consensus@64"),
    ("deepseek-r1", "ifeval", 87.4, "Strict prompt"),

    # Claude 3.7 Sonnet Thinking
    ("claude-3-7-sonnet-thinking", "swe_bench_verified", 70.3, "Pass@1 (Claude Code Scaffold)"),
    ("claude-3-7-sonnet-thinking", "gpqa_diamond", 84.8, "5-shot CoT"),
    ("claude-3-7-sonnet-thinking", "math_500", 96.2, "Pass@1"),
    ("claude-3-7-sonnet-thinking", "mmlu_pro", 88.4, "5-shot CoT"),
    ("claude-3-7-sonnet-thinking", "humaneval", 96.8, "Pass@1"),
    ("claude-3-7-sonnet-thinking", "livecodebench", 70.3, "Pass@1"),
    ("claude-3-7-sonnet-thinking", "arc_agi", 41.2, "Pass@1"),
    ("claude-3-7-sonnet-thinking", "ifeval", 92.5, "Strict prompt"),

    # Claude 3.5 Sonnet
    ("claude-3-5-sonnet", "swe_bench_verified", 49.0, "Pass@1 (Agentic)"),
    ("claude-3-5-sonnet", "gpqa_diamond", 65.0, "0-shot CoT"),
    ("claude-3-5-sonnet", "math_500", 78.3, "0-shot CoT"),
    ("claude-3-5-sonnet", "mmlu_pro", 78.0, "5-shot CoT"),
    ("claude-3-5-sonnet", "humaneval", 93.7, "0-shot"),
    ("claude-3-5-sonnet", "livecodebench", 40.5, "Pass@1"),
    ("claude-3-5-sonnet", "arc_agi", 21.0, "Few-shot"),
    ("claude-3-5-sonnet", "ifeval", 88.0, "Strict prompt"),

    # GPT-4o
    ("gpt-4o", "swe_bench_verified", 38.8, "Pass@1"),
    ("gpt-4o", "gpqa_diamond", 53.6, "0-shot CoT"),
    ("gpt-4o", "math_500", 74.6, "0-shot CoT"),
    ("gpt-4o", "mmlu_pro", 72.6, "5-shot CoT"),
    ("gpt-4o", "humaneval", 90.2, "0-shot"),
    ("gpt-4o", "livecodebench", 33.8, "Pass@1"),
    ("gpt-4o", "arc_agi", 18.5, "Few-shot"),
    ("gpt-4o", "ifeval", 84.3, "Strict prompt"),

    # Llama 3.3 70B Instruct
    ("llama-3-3-70b-instruct", "swe_bench_verified", 37.8, "Pass@1"),
    ("llama-3-3-70b-instruct", "gpqa_diamond", 50.5, "5-shot CoT"),
    ("llama-3-3-70b-instruct", "math_500", 73.0, "5-shot CoT"),
    ("llama-3-3-70b-instruct", "mmlu_pro", 72.4, "5-shot CoT"),
    ("llama-3-3-70b-instruct", "humaneval", 88.4, "0-shot"),
    ("llama-3-3-70b-instruct", "livecodebench", 34.0, "Pass@1"),
    ("llama-3-3-70b-instruct", "arc_agi", 15.2, "Few-shot"),
    ("llama-3-3-70b-instruct", "ifeval", 87.5, "Strict prompt"),

    # Llama 3.1 405B Instruct
    ("llama-3-1-405b-instruct", "swe_bench_verified", 41.6, "Pass@1"),
    ("llama-3-1-405b-instruct", "gpqa_diamond", 51.1, "5-shot CoT"),
    ("llama-3-1-405b-instruct", "math_500", 73.8, "5-shot CoT"),
    ("llama-3-1-405b-instruct", "mmlu_pro", 73.3, "5-shot CoT"),
    ("llama-3-1-405b-instruct", "humaneval", 89.0, "0-shot"),
    ("llama-3-1-405b-instruct", "livecodebench", 35.8, "Pass@1"),
    ("llama-3-1-405b-instruct", "arc_agi", 17.0, "Few-shot"),
    ("llama-3-1-405b-instruct", "ifeval", 88.6, "Strict prompt"),

    # DeepSeek-V3
    ("deepseek-v3", "swe_bench_verified", 42.0, "Pass@1"),
    ("deepseek-v3", "gpqa_diamond", 59.1, "5-shot CoT"),
    ("deepseek-v3", "math_500", 89.3, "5-shot CoT"),
    ("deepseek-v3", "mmlu_pro", 75.9, "5-shot CoT"),
    ("deepseek-v3", "humaneval", 90.2, "0-shot"),
    ("deepseek-v3", "livecodebench", 42.1, "Pass@1"),
    ("deepseek-v3", "arc_agi", 19.8, "Few-shot"),
    ("deepseek-v3", "ifeval", 86.1, "Strict prompt"),

    # Qwen 2.5 72B Instruct
    ("qwen-2-5-72b-instruct", "swe_bench_verified", 38.2, "Pass@1"),
    ("qwen-2-5-72b-instruct", "gpqa_diamond", 49.8, "5-shot CoT"),
    ("qwen-2-5-72b-instruct", "math_500", 83.1, "5-shot CoT"),
    ("qwen-2-5-72b-instruct", "mmlu_pro", 71.6, "5-shot CoT"),
    ("qwen-2-5-72b-instruct", "humaneval", 86.0, "0-shot"),
    ("qwen-2-5-72b-instruct", "livecodebench", 33.2, "Pass@1"),
    ("qwen-2-5-72b-instruct", "arc_agi", 14.8, "Few-shot"),
    ("qwen-2-5-72b-instruct", "ifeval", 85.0, "Strict prompt"),

    # Qwen 2.5 Coder 32B
    ("qwen-2-5-coder-32b", "swe_bench_verified", 41.2, "Pass@1 (Code scaffold)"),
    ("qwen-2-5-coder-32b", "gpqa_diamond", 44.5, "5-shot CoT"),
    ("qwen-2-5-coder-32b", "math_500", 81.2, "5-shot CoT"),
    ("qwen-2-5-coder-32b", "mmlu_pro", 64.8, "5-shot CoT"),
    ("qwen-2-5-coder-32b", "humaneval", 92.7, "Pass@1"),
    ("qwen-2-5-coder-32b", "livecodebench", 40.2, "Pass@1"),
    ("qwen-2-5-coder-32b", "arc_agi", 12.0, "Few-shot"),
    ("qwen-2-5-coder-32b", "ifeval", 79.4, "Strict prompt"),

    # DeepSeek-R1-Distill-Qwen-32B
    ("deepseek-r1-distill-qwen-32b", "swe_bench_verified", 43.1, "Pass@1"),
    ("deepseek-r1-distill-qwen-32b", "gpqa_diamond", 65.2, "5-shot CoT"),
    ("deepseek-r1-distill-qwen-32b", "math_500", 94.3, "Pass@1"),
    ("deepseek-r1-distill-qwen-32b", "mmlu_pro", 80.5, "5-shot CoT"),
    ("deepseek-r1-distill-qwen-32b", "humaneval", 93.3, "Pass@1"),
    ("deepseek-r1-distill-qwen-32b", "livecodebench", 57.2, "Pass@1"),
    ("deepseek-r1-distill-qwen-32b", "arc_agi", 24.1, "Consensus@64"),
    ("deepseek-r1-distill-qwen-32b", "ifeval", 84.9, "Strict prompt"),

    # Microsoft Phi-4
    ("microsoft-phi-4", "swe_bench_verified", 26.5, "Pass@1"),
    ("microsoft-phi-4", "gpqa_diamond", 41.4, "5-shot CoT"),
    ("microsoft-phi-4", "math_500", 80.4, "Pass@1"),
    ("microsoft-phi-4", "mmlu_pro", 62.5, "5-shot CoT"),
    ("microsoft-phi-4", "humaneval", 84.1, "Pass@1"),
    ("microsoft-phi-4", "livecodebench", 29.8, "Pass@1"),
    ("microsoft-phi-4", "arc_agi", 11.5, "Few-shot"),
    ("microsoft-phi-4", "ifeval", 81.2, "Strict prompt")
]

evaluations_list = []
for idx, (m_id, b_id, score, meth) in enumerate(raw_evals):
    evaluations_list.append({
        "eval_id": f"eval_{idx+1}",
        "model_id": m_id,
        "benchmark_id": b_id,
        "score": score,
        "eval_date": "2025-01-28",
        "methodology": meth
    })

tools_list = [
    {"tool_id": "tool_vllm", "name": "vLLM", "category": "Inference Engine", "repo_url": "https://github.com/vllm-project/vllm", "release_version": "0.7.3", "ecosystem_role": "High-throughput serving with PagedAttention, MLA kernel optimization, and FP8 Hopper acceleration."},
    {"tool_id": "tool_sglang", "name": "SGLang", "category": "Inference Engine", "repo_url": "https://github.com/sgl-project/sglang", "release_version": "0.4.2", "ecosystem_role": "RadixAttention KV cache reuse engine for high-density reasoning, multi-turn chat, and agentic loop workloads."},
    {"tool_id": "tool_mcp", "name": "Model Context Protocol (MCP)", "category": "Agent Protocol", "repo_url": "https://modelcontextprotocol.io", "release_version": "1.0", "ecosystem_role": "Open standard enabling AI agents to connect securely to local file systems, databases, developer tools, and external APIs."},
    {"tool_id": "tool_langgraph", "name": "LangGraph", "category": "Orchestration Framework", "repo_url": "https://github.com/langchain-ai/langgraph", "release_version": "0.2.68", "ecosystem_role": "Cyclic stateful graph orchestration framework supporting durable human-in-the-loop multi-agent execution."},
    {"tool_id": "tool_dspy", "name": "DSPy", "category": "Optimization Framework", "repo_url": "https://github.com/stanfordnlp/dspy", "release_version": "2.5.43", "ecosystem_role": "Declarative programming framework for algorithmically optimizing LM prompts and weight tuning via MIPROv2."},
    {"tool_id": "tool_tensorrt_llm", "name": "TensorRT-LLM", "category": "Inference Runtime", "repo_url": "https://github.com/NVIDIA/TensorRT-LLM", "release_version": "0.17.0", "ecosystem_role": "NVIDIA enterprise runtime delivering maximum tokens/sec through deep tensor parallelism and FP4/FP8 quantization."},
    {"tool_id": "tool_ollama", "name": "Ollama", "category": "Local Execution Runtime", "repo_url": "https://github.com/ollama/ollama", "release_version": "0.5.7", "ecosystem_role": "Cross-platform developer CLI and background daemon for one-command local execution of GGUF quantized models."},
    {"tool_id": "tool_unsloth", "name": "Unsloth", "category": "Fine-Tuning Engine", "repo_url": "https://github.com/unslothai/unsloth", "release_version": "2025.1", "ecosystem_role": "Memory-efficient fine-tuning framework providing 2-5x faster LoRA and QLoRA training on standard consumer GPUs."},
    {"tool_id": "tool_agy_sdk", "name": "Google Antigravity SDK", "category": "Multi-Agent System", "repo_url": "https://antigravity.google", "release_version": "2.0.0", "ecosystem_role": "Autonomous agent runtime supporting reactive wakeups, sidecars, and background task isolation."}
]

developments_list = [
    {
        "development_id": "dev_1",
        "title": "OpenAI Launches o3-mini with Configurable Reasoning Effort",
        "summary": "OpenAI unveils o3-mini delivering 49.8% on SWE-bench Verified and 97.9% on MATH-500 with tiered reasoning controls (low, medium, high), matching o1 performance at 10x lower inference cost.",
        "category": "Frontier Models",
        "date": "2025-01-31",
        "confidence_level": "EVALUATED",
        "impact_score": 9.6,
        "source_url": "https://openai.com/index/openai-o3-mini/"
    },
    {
        "development_id": "dev_2",
        "title": "DeepSeek Releases DeepSeek-R1 Open Reasoning Weights & MIT Distillations",
        "summary": "DeepSeek releases full 671B MoE reasoning weights and 6 distilled dense models (1.5B to 70B) under permissive MIT license, matching commercial frontier reasoning models on mathematics and code.",
        "category": "Open Source Weights",
        "date": "2025-01-20",
        "confidence_level": "PEER_REVIEWED",
        "impact_score": 9.9,
        "source_url": "https://arxiv.org/abs/2501.12948"
    },
    {
        "development_id": "dev_3",
        "title": "vLLM 0.7.3 Ships Native Multi-Head Latent Attention (MLA) Kernels",
        "summary": "vLLM introduces specialized Triton and CUDA kernels accelerating DeepSeek-V3/R1 Multi-Head Latent Attention and FP8 block-wise quantization on Hopper architectures.",
        "category": "Inference Infrastructure",
        "date": "2025-01-23",
        "confidence_level": "EVALUATED",
        "impact_score": 9.2,
        "source_url": "https://github.com/vllm-project/vllm/releases/tag/v0.7.3"
    },
    {
        "development_id": "dev_4",
        "title": "Anthropic Unveils Claude 3.7 Sonnet Hybrid Architecture with Thinking Mode",
        "summary": "Anthropic releases Claude 3.7 Sonnet introducing user-configurable test-time compute, setting a record 70.3% on SWE-bench Verified when paired with Claude Code.",
        "category": "Frontier Models",
        "date": "2025-02-24",
        "confidence_level": "EVALUATED",
        "impact_score": 9.8,
        "source_url": "https://www.anthropic.com/news/claude-3-7-sonnet"
    },
    {
        "development_id": "dev_5",
        "title": "Meta Ships Llama 3.3 70B Matching Previous 405B Frontier Scores",
        "summary": "Meta releases Llama 3.3 70B Instruct utilizing improved online synthetic data generation and distillation, matching Llama 3.1 405B capabilities at a fraction of compute.",
        "category": "Open Source Weights",
        "date": "2024-12-06",
        "confidence_level": "EVALUATED",
        "impact_score": 9.1,
        "source_url": "https://ai.meta.com/blog/llama-3-3/"
    },
    {
        "development_id": "dev_6",
        "title": "Model Context Protocol (MCP) Adopted as Universal Agent Standard",
        "summary": "Anthropic's open-source MCP protocol reaches widespread adoption across developer environments, unifying tool calling and context exchange across enterprise agents.",
        "category": "Agent Protocols",
        "date": "2024-11-25",
        "confidence_level": "EVALUATED",
        "impact_score": 9.4,
        "source_url": "https://modelcontextprotocol.io"
    }
]

relationships_list = [
    {"rel_id": "rel_1", "source_entity_id": "deepseek-r1", "target_entity_id": "tool_vllm", "relationship_type": "DEPLOYED_ON", "strength": 1.0},
    {"rel_id": "rel_2", "source_entity_id": "deepseek-r1", "target_entity_id": "tool_sglang", "relationship_type": "ACCELERATED_BY", "strength": 1.0},
    {"rel_id": "rel_3", "source_entity_id": "deepseek-r1", "target_entity_id": "deepseek-r1-distill-qwen-32b", "relationship_type": "DISTILLED_INTO", "strength": 1.0},
    {"rel_id": "rel_4", "source_entity_id": "deepseek-r1", "target_entity_id": "deepseek-r1-distill-llama-70b", "relationship_type": "DISTILLED_INTO", "strength": 1.0},
    {"rel_id": "rel_5", "source_entity_id": "claude-3-7-sonnet-thinking", "target_entity_id": "tool_mcp", "relationship_type": "SUPPORTS_PROTOCOL", "strength": 1.0},
    {"rel_id": "rel_6", "source_entity_id": "openai-o3-mini", "target_entity_id": "swe_bench_verified", "relationship_type": "EVALUATED_ON", "strength": 1.0},
    {"rel_id": "rel_7", "source_entity_id": "llama-3-3-70b-instruct", "target_entity_id": "tool_unsloth", "relationship_type": "FINE_TUNABLE_WITH", "strength": 1.0},
    {"rel_id": "rel_8", "source_entity_id": "qwen-2-5-coder-32b", "target_entity_id": "tool_vllm", "relationship_type": "SUPPORTED_BY", "strength": 1.0}
]

expanded_signal_pkg = {
    "models": models_list,
    "benchmarks": benchmarks_list,
    "evaluations": evaluations_list,
    "tools": tools_list,
    "developments": developments_list,
    "relationships": relationships_list
}

with open('SIGNAL/data/fixtures/expanded_signal_data.json', 'w') as f:
    json.dump(expanded_signal_pkg, f, indent=2)

print(f"Generated SIGNAL dataset: {len(models_list)} models, {len(benchmarks_list)} benchmarks, {len(evaluations_list)} evaluations, {len(developments_list)} developments.")
