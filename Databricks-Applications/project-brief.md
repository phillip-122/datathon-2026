# Datathon: Two Independent Intelligence Products
## Master project brief and Antigravity execution plan

**Mission:** Build two independently demonstrable applications that turn fragmented, evolving information into structured, searchable, explainable knowledge using Databricks.

**Projects:**
1. AI Landscape Intelligence
2. Food Intelligence

**Execution strategy:** Parallel development, independent repositories or project directories, separate data models, separate Genie Agents, separate application implementations.

**Primary constraint:** Both applications must make meaningful progress without Databricks access. Do not block application development on workspace availability.

---

# PART I — PRODUCT PITCHES AND DEMO END GOALS

# PROJECT A — AI LANDSCAPE INTELLIGENCE

## Working name: SIGNAL

### The problem

AI is progressing across research, models, tools, infrastructure, and applications, but information about that progression is scattered.

A new model release might appear on a company's blog. Its technical capabilities are described in a research paper. Its actual performance appears in benchmarks. Its implementation appears in a GitHub repository. Other tools subsequently integrate it.

These developments are related, but the information is disconnected.

People can follow AI news, but following news does not necessarily mean understanding the AI landscape.

**The product opportunity:** Build an evolving intelligence system that tracks the development of AI technologies and helps users understand what is new, what is important, what has changed, and how developments relate to each other.

### The product

SIGNAL is an AI landscape observatory.

It continuously collects developments from multiple sources, extracts structured information, connects related technologies, and maintains a historical representation of the AI ecosystem.

It is not another news feed, search engine, or chatbot.

Its distinguishing capability is understanding the relationships between developments.

### What it tracks

| Domain | Information |
|---|---|
| Models | Releases, versions, providers, architectures |
| Capabilities | Reasoning, coding, vision, agents, multimodal capabilities |
| Research | Papers, methods, techniques, findings |
| Benchmarks | Evaluations, scores, methodologies, limitations |
| Tools | Frameworks, agents, developer tools, applications |
| Infrastructure | Hardware, inference, memory, orchestration |
| Ecosystem | Integrations, dependencies, adoption, alternatives |

### The killer demonstration

**Question: What has actually changed in the AI agent ecosystem over the last 30 days?**

The application:

1. Displays a timeline of recently discovered developments.
2. Groups related developments into themes.
3. Connects research developments to models, tools, and applications.
4. Distinguishes announcements from independently evaluated results.
5. Identifies technologies gaining momentum and technologies becoming less relevant.
6. Lets the user investigate a particular development and trace its supporting sources.

### The final demo experience

The judge opens SIGNAL.

They see:

- A live-ish feed of newly ingested AI developments.
- An interactive ecosystem map.
- A timeline showing how technologies and capabilities have changed.
- A comparison view for models or tools.
- An investigation interface powered by Genie.

The presenter introduces a new source document or release record.

The system ingests it, extracts entities and relationships, updates the relevant structured records, and makes the new information available to the application.

The presenter then asks:

"Which existing tools and capabilities does this development affect?"

The application shows related developments and source evidence.

**The wow moment:** A new development changes the system's understanding of an existing technology landscape rather than merely appearing as another news article.

### What makes this useful?

- Researchers can explore emerging techniques.
- Developers can understand which tools and frameworks are relevant.
- Students can understand how the AI ecosystem is evolving.
- Technical decision-makers can compare technologies and trace claims back to evidence.

### Explicit non-goals

- Not a generic AI chatbot.
- Not a simple AI news aggregator.
- Not an unverified ranking of the "best AI tools."
- Not an autonomous system making unsupported predictions about which companies will win.
- Not a scraped collection of articles with summaries.

---

# PROJECT B — FOOD INTELLIGENCE

## Working name: FOODMECHANICS

### The problem

Food knowledge is scattered across nutritional databases, scientific publications, recipes, product labels, culinary references, and food chemistry research.

These sources describe different aspects of the same food system.

A nutrition database can tell you how much protein an ingredient contains. A food chemistry paper might explain how its proteins interact with water or air. A recipe demonstrates how that ingredient behaves under certain conditions.

But connecting those pieces requires substantial manual research.

**The product opportunity:** Build an evidence-grounded food intelligence system that connects nutrition, ingredients, chemistry, processing, flavor, and culinary outcomes.

### The product

FOODMECHANICS is an interactive food science investigation platform.

It allows users to explore why food behaves the way it does, how ingredient properties interact, and what happens when recipes or processes change.

It is not simply a recipe generator or calorie tracker.

Its distinguishing capability is connecting the underlying mechanisms of food to observable outcomes.

### What it tracks

| Domain | Information |
|---|---|
| Ingredients | Composition, origin, functionality |
| Nutrition | Calories, protein, carbohydrates, fats, fiber, micronutrients |
| Chemistry | Proteins, starches, fats, emulsions, foams, gels |
| Processing | Temperature, mixing, hydration, cooking, cooling |
| Flavor | Flavor compounds, pairings, sensory characteristics |
| Recipes | Ingredients, quantities, processes, outcomes |
| Experiments | Observations, substitutions, measurements |
| Research | Scientific findings, mechanisms, limitations |

### The killer demonstration

**Question: How can we make a lower-fat whipped dessert while preserving volume and structural stability?**

The application investigates:

1. The structural differences between egg-white foam, cream foam, and milk-based systems.
2. The role of fat, proteins, water, sugar, and stabilizers.
3. How different ingredients affect foam formation and stability.
4. What scientific evidence supports each mechanism.
5. Which substitutions could be worth experimentally testing.
6. How the resulting formulation compares nutritionally.

The system should distinguish established scientific mechanisms from hypotheses about a particular recipe.

### The final demo experience

The judge opens FOODMECHANICS.

They see:

- An ingredient explorer.
- A nutritional composition comparison.
- An interactive ingredient-property relationship map.
- A food chemistry investigation interface.
- A source-backed explanation of mechanisms.
- A recipe or formulation comparison.

The presenter selects an ingredient, such as egg white.

The system displays its nutritional composition and relevant functional properties.

The presenter explores its relationships to foam formation, protein structure, and stabilization.

Then the presenter asks:

"What would change if I substituted some of the egg white with milk?"

The system investigates the available evidence, distinguishes known mechanisms from uncertain outcomes, and presents the expected trade-offs.

**The wow moment:** The application connects nutrition, chemistry, and culinary behavior into one investigation instead of returning an isolated recipe or nutrition label.

### What makes this useful?

- Home cooks can understand ingredient substitutions.
- People interested in nutrition can understand more than macronutrient labels.
- Bakers can investigate texture and structure.
- Food enthusiasts can explore flavor and food science.
- Students can investigate food chemistry using accessible examples.

### Explicit non-goals

- Not another calorie tracker.
- Not another recipe website.
- Not an AI recipe generator pretending to know food chemistry.
- Not a medical or dietary prescription system.
- Not a scientific authority that presents unsupported predictions as established fact.

---

# PART II — DATA AND DATABRICKS ARCHITECTURE

Each project must have its own independent implementation.

Suggested directory structure:

```text
datathon/
│
├── SIGNAL/
│   ├── app/
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── fixtures/
│   ├── ingestion/
│   ├── pipelines/
│   ├── analytics/
│   ├── genie/
│   ├── evaluation/
│   ├── docs/
│   └── README.md
│
├── FOODMECHANICS/
│   ├── app/
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── fixtures/
│   ├── ingestion/
│   ├── pipelines/
│   ├── analytics/
│   ├── genie/
│   ├── evaluation/
│   ├── docs/
│   └── README.md
│
└── coordination/
    ├── MASTER_PLAN.md
    ├── AGENT_ASSIGNMENTS.md
    ├── INTEGRATION_CONTRACT.md
    └── PROGRESS.md
```

## Common architectural principles

Both projects should follow a medallion-style data architecture.

### Bronze — raw evidence

Store the original incoming records.

Examples:

- JSON API responses.
- RSS feed records.
- Research metadata.
- Product and ingredient records.
- Scientific publication metadata.
- Recipe records.

Requirements:

- Preserve source URLs.
- Preserve original publication or release timestamps.
- Record ingestion timestamps.
- Record source identifiers.
- Preserve the original content or raw payload.
- Never silently discard an incoming record.

### Silver — normalized knowledge

Convert source-specific information into consistent records.

Examples:

- A model release becomes a normalized model-release record.
- A research paper becomes a publication record.
- An ingredient becomes a normalized ingredient entity.
- A nutritional measurement receives a standardized unit and serving basis.
- A scientific claim becomes an evidence-linked claim.

Requirements:

- Deduplicate.
- Normalize identifiers.
- Validate fields.
- Preserve provenance.
- Track missing and uncertain values.
- Maintain relationships between entities.
- Handle changed source records without losing historical context.

### Gold — application-ready information

Create focused datasets for the user-facing applications.

SIGNAL:
- Recent developments.
- Model and tool comparisons.
- Capability timelines.
- Research-to-tool relationships.
- Emerging themes.
- Source-backed development histories.

FOODMECHANICS:
- Ingredient comparisons.
- Nutritional composition.
- Functional ingredient properties.
- Mechanism-to-outcome relationships.
- Recipe formulations.
- Evidence-backed food science investigations.

## Incremental ingestion requirements

Do not repeatedly reprocess every source.

Implement:

1. Source-specific ingestion timestamps.
2. Stable source identifiers.
3. Content hashes where useful.
4. Incremental discovery.
5. Duplicate detection.
6. Retry-safe processing.
7. Update detection.
8. Historical version preservation.

For file-based sources, investigate Databricks Auto Loader.

For API-based sources, implement a source-specific incremental polling mechanism with checkpoints.

Use continuous streaming only where the source and problem justify low latency. A daily or hourly update does not automatically require a continuously running stream.

The architecture should demonstrate why new information can be processed without rebuilding the entire dataset.

---

# PART III — GENIE AGENT CONFIGURATION GUIDES

Create TWO separate Genie Agents.

1. SIGNAL Intelligence
2. FOODMECHANICS Intelligence

Do not combine them into one Genie Agent.

Each Genie Agent should operate on its project's curated Unity Catalog datasets.

Important: Genie Agents are query interfaces over structured data. The raw documents should be ingested and transformed into structured records before they become the primary basis for Genie queries.

## A. SIGNAL GENIE GUIDE

### Purpose

SIGNAL Intelligence helps users investigate the AI ecosystem through structured information about models, tools, research, benchmarks, capabilities, and developments.

### Proposed datasets

| Table or view | Purpose |
|---|---|
| `developments` | Individual announcements and discoveries |
| `models` | Model entities and versions |
| `tools` | Frameworks, applications, and developer tools |
| `research_publications` | Papers and research metadata |
| `benchmarks` | Evaluation records and methodologies |
| `capabilities` | Normalized capability taxonomy |
| `development_entities` | Relationships between developments and entities |
| `evidence` | Source references and supporting claims |
| `development_history` | Historical changes and observations |
| `v_recent_developments` | Recent application-ready developments |
| `v_model_comparisons` | Comparable model evaluation records |
| `v_ecosystem_relationships` | Connected models, tools, research, and capabilities |

### Genie general instructions

Copy and adapt the following as the General Instructions:

```text
You are SIGNAL Intelligence, an analytical assistant for exploring
the evolving AI technology ecosystem.

Your purpose is to help users understand developments, technologies,
capabilities, research, benchmarks, and relationships across the
AI landscape.

DATA RULES:

1. Base factual answers about tracked developments on the available
   datasets and their supporting source records.

2. Distinguish announcements, research claims, independently
   evaluated results, and community observations.

3. Do not treat an announced capability as independently verified.

4. Do not compare benchmark scores unless the benchmark,
   evaluation methodology, model version, and relevant conditions
   are sufficiently comparable.

5. Distinguish publication dates, release dates, discovery dates,
   and ingestion dates.

6. Do not assume that missing records indicate that a technology
   does not exist.

7. Preserve source attribution and identify the supporting sources
   for important claims.

8. When describing trends, specify the time period and available
   evidence. Avoid claiming momentum from a small number of records.

9. Distinguish a direct relationship from a speculative relationship.

10. Do not invent model capabilities, release dates, benchmark
    results, tool integrations, or adoption statistics.

ANALYTICAL BEHAVIOR:

- Explain why developments are related.
- Compare versions and technologies when comparable evidence exists.
- Identify meaningful changes over time.
- Identify conflicting claims.
- Explain what additional evidence would be needed to establish
  a conclusion.
- Prefer structured comparisons over generic summaries.

If a question cannot be answered from the available datasets,
state what information is missing rather than fabricating an answer.
```

### Recommended Genie questions

1. What are the most significant AI developments in the last 30 days?
2. Which capabilities have received the most new research attention?
3. Which new tools are related to recent model releases?
4. Compare these two models using comparable benchmarks.
5. Which tools depend on or integrate with a particular model?
6. Which developments are announcements versus independently evaluated results?
7. What changed in the AI agent ecosystem over the last month?
8. Which technologies have had the largest number of related developments?
9. What are the most important research-to-product connections?
10. What claims in the current dataset have conflicting evidence?

### Example SQL assets to configure

Provide Genie with verified example queries for:

- Most recent developments by category.
- Developments grouped by capability.
- Model comparison by benchmark and version.
- Research publications connected to specific tools.
- Development counts by week.
- Source evidence for a particular development.
- Historical changes to a model or tool.

Do not expect text instructions alone to teach complicated SQL logic. Build and test the SQL queries against the actual tables first.

---

## B. FOODMECHANICS GENIE GUIDE

### Purpose

FOODMECHANICS Intelligence helps users investigate food composition, food chemistry, ingredient functionality, processing, flavor, and culinary outcomes.

### Proposed datasets

| Table or view | Purpose |
|---|---|
| `ingredients` | Normalized ingredient entities |
| `nutritional_profiles` | Nutrient composition and serving basis |
| `functional_properties` | Ingredient properties and functionality |
| `chemical_compounds` | Relevant compounds and characteristics |
| `food_mechanisms` | Scientific mechanisms |
| `mechanism_relationships` | Relationships between mechanisms and properties |
| `recipes` | Recipe records |
| `recipe_ingredients` | Ingredient quantities and proportions |
| `processing_methods` | Temperature, mixing, hydration, and other processes |
| `food_experiments` | Experimental observations |
| `scientific_evidence` | Supporting publications and claims |
| `v_ingredient_comparisons` | Comparable ingredient profiles |
| `v_recipe_nutrition` | Calculated recipe nutrition |
| `v_mechanism_evidence` | Mechanisms linked to scientific evidence |

### Genie general instructions

```text
You are FOODMECHANICS Intelligence, an analytical assistant for
investigating food science, nutrition, ingredients, chemistry,
flavor, and culinary behavior.

Your purpose is to connect food composition and scientific
mechanisms to observable culinary properties and outcomes.

DATA RULES:

1. Distinguish measured nutritional values from estimated values.

2. Always consider serving basis and units when comparing
   nutritional composition.

3. Do not compare values measured on different bases without
   explicitly normalizing them.

4. Distinguish an ingredient's nutritional composition from
   its functional properties.

5. Distinguish scientific mechanisms from recipe-specific
   observations and hypotheses.

6. Link scientific explanations to their supporting evidence.

7. Do not treat an association as proof of a mechanism.

8. Distinguish established findings from uncertain or
   incompletely studied effects.

9. Do not invent chemical properties, nutritional values,
   ingredient interactions, or experimental outcomes.

10. When discussing substitutions, explain the possible trade-offs
    in structure, flavor, nutrition, and processing.

ANALYTICAL BEHAVIOR:

- Explain why ingredients behave differently.
- Compare nutritional composition on consistent serving bases.
- Connect ingredient properties to relevant food mechanisms.
- Explain how processing conditions can change outcomes.
- Identify which claims are supported by evidence.
- Identify uncertainties and recommend what would need
  to be experimentally tested.

When discussing a proposed formulation, clearly distinguish
expected behavior from experimentally verified results.

Do not provide medical or individualized dietary prescriptions.
```

### Recommended Genie questions

1. Compare the nutritional composition of these ingredients.
2. What properties make egg whites useful for foam formation?
3. Explain the difference between fat-based and protein-based foams.
4. What mechanisms contribute to foam stability?
5. How might replacing egg white with milk change a formulation?
6. Which ingredients contain proteins with relevant functional properties?
7. What scientific evidence supports this ingredient interaction?
8. How does cooking temperature affect the properties of this ingredient?
9. Compare two formulations by nutrition and ingredient functionality.
10. Which food science questions in this dataset have limited evidence?

### Example SQL assets to configure

Provide verified examples for:

- Nutritional comparisons normalized per 100 grams.
- Ingredient composition by nutrient.
- Ingredient-to-property relationships.
- Mechanisms connected to supporting publications.
- Recipe nutritional calculations.
- Ingredients associated with a specified food mechanism.
- Experimental outcomes grouped by processing conditions.

Do not imply that a relationship in the knowledge graph proves a successful substitution.

---

# PART IV — ANTIGRAVITY MULTI-AGENT EXECUTION PLAN

## Mission for the orchestrator

You are the engineering lead for two independent Databricks applications.

Your job is to coordinate parallel development, delegate clearly scoped tasks, maintain integration contracts, and deliver working applications.

You must not wait for Databricks access before beginning useful work.

Do not combine the projects into one application.

Do not let one project's development block the other.

Do not let agents independently invent incompatible schemas or interfaces.

Create a master progress tracker and require agents to report:
- Files created or changed.
- Decisions made.
- Data contracts established.
- Tests performed.
- Remaining blockers.
- Dependencies on other agents.

## Agent structure

Use one orchestration agent and eight implementation agents.

### Orchestrator — Project Director

Responsibilities:
- Create the project directories.
- Assign independent tasks.
- Establish shared coding conventions.
- Review data contracts.
- Resolve conflicting implementation decisions.
- Track dependencies.
- Integrate completed work.
- Ensure both projects remain independently runnable.
- Prevent duplicated work.
- Maintain the master progress file.

The orchestrator should not implement every feature itself.

---

## SIGNAL agent team

### Agent S1 — Domain Research and Data Acquisition

Mission: Build the AI landscape data acquisition layer.

Investigate feasible sources:

- arXiv API or feeds.
- Hugging Face model and dataset APIs.
- GitHub repository and release APIs.
- Official AI company release feeds.
- Public benchmark datasets and documentation.

Tasks:
1. Identify available APIs and feeds.
2. Record access requirements, quotas, and licensing considerations.
3. Implement at least two working ingestion sources.
4. Define source-specific identifiers and timestamps.
5. Store original source payloads.
6. Create fixture datasets for offline development.
7. Document limitations and missing fields.

Deliverables:
- `ingestion/sources/`
- `data/raw/`
- `data/fixtures/`
- `docs/SOURCES.md`
- `docs/DATA_DICTIONARY.md`

Do not build a web scraper that depends on fragile HTML selectors when an official API or feed exists.

### Agent S2 — SIGNAL Data Engineering

Mission: Build the normalized data and incremental processing pipeline.

Tasks:
1. Define Bronze, Silver, and Gold schemas.
2. Normalize models, tools, publications, benchmarks, and developments.
3. Implement deduplication.
4. Implement incremental ingestion checkpoints.
5. Track changed records.
6. Preserve historical versions.
7. Build application-ready analytical views.
8. Add data-quality checks.

Deliverables:
- `pipelines/`
- `analytics/`
- `docs/SCHEMA.md`
- `tests/`

Do not assume that every new article is a new technology or that every updated record is a new development.

### Agent S3 — SIGNAL Application

Mission: Implement the actual user-facing application independently of Databricks availability.

Build:

1. Overview dashboard.
2. Recent developments timeline.
3. Search and filtering.
4. Model and tool comparison.
5. Ecosystem relationship visualization.
6. Development detail pages with source evidence.
7. Genie investigation interface.
8. Loading, empty, and error states.

Deliverables:
- `app/`
- `README.md`
- Application screenshots.
- Local run instructions.

The app must work against local fixture data before Databricks is connected.

### Agent S4 — SIGNAL Genie and Evaluation

Mission: Design and validate the SIGNAL Genie Agent.

Tasks:
1. Review the proposed table schemas.
2. Create Genie configuration documentation.
3. Write general instructions.
4. Prepare example SQL queries.
5. Define semantic terms and relationships.
6. Create a question-and-answer evaluation set.
7. Test answers against known expected results when a Genie Agent is available.
8. Identify incorrect joins, misleading metrics, and unsupported claims.

Deliverables:
- `genie/GENIE_GUIDE.md`
- `genie/EXAMPLE_QUERIES.sql`
- `evaluation/QUESTIONS.json`
- `evaluation/EXPECTED_RESULTS.md`

Until Databricks is available, validate the underlying SQL locally where possible.

---

## FOODMECHANICS agent team

### Agent F1 — Food Science and Data Acquisition

Mission: Identify and implement reliable food information sources.

Investigate:

- USDA FoodData Central.
- Open Food Facts.
- PubChem.
- Crossref or OpenAlex publication metadata.
- Public scientific food science resources.
- Carefully curated recipe and experimental datasets.

Tasks:
1. Identify feasible sources.
2. Record API requirements and licenses.
3. Implement at least two real data sources.
4. Create structured ingredient records.
5. Preserve nutritional measurement units and serving bases.
6. Collect scientific publication metadata.
7. Build fixture datasets for offline development.
8. Document which properties are measured, derived, or inferred.

Deliverables:
- `ingestion/sources/`
- `data/raw/`
- `data/fixtures/`
- `docs/SOURCES.md`
- `docs/DATA_DICTIONARY.md`

Do not invent chemical properties or fabricate scientific findings to make the dataset look comprehensive.

### Agent F2 — FOODMECHANICS Data Engineering

Mission: Build the food knowledge model and incremental pipelines.

Tasks:
1. Define ingredient, nutrient, compound, mechanism, and recipe schemas.
2. Normalize ingredient identifiers.
3. Normalize nutrient measurements.
4. Preserve serving bases and measurement methods.
5. Build ingredient-property relationships.
6. Build mechanism-to-evidence relationships.
7. Create recipe nutrition calculations.
8. Add validation and provenance checks.

Deliverables:
- `pipelines/`
- `analytics/`
- `docs/SCHEMA.md`
- `tests/`

Avoid treating every ingredient-property relationship as scientifically proven.

### Agent F3 — FOODMECHANICS Application

Mission: Build the actual food science application independently of Databricks availability.

Build:

1. Ingredient explorer.
2. Nutritional comparison interface.
3. Functional properties explorer.
4. Food mechanism relationship map.
5. Recipe formulation comparison.
6. Scientific evidence detail pages.
7. Genie investigation interface.
8. Loading, empty, and error states.

Deliverables:
- `app/`
- `README.md`
- Application screenshots.
- Local run instructions.

The app must be usable with local fixture data.

### Agent F4 — FOODMECHANICS Genie and Evaluation

Mission: Design and validate the FOODMECHANICS Genie Agent.

Tasks:
1. Review the food data schema.
2. Create Genie configuration documentation.
3. Write general instructions.
4. Prepare example SQL queries.
5. Define nutrition and food science terminology.
6. Build a question-and-answer evaluation set.
7. Validate nutritional comparisons.
8. Test scientific claim provenance.
9. Identify hallucinations and unsupported conclusions.

Deliverables:
- `genie/GENIE_GUIDE.md`
- `genie/EXAMPLE_QUERIES.sql`
- `evaluation/QUESTIONS.json`
- `evaluation/EXPECTED_RESULTS.md`

---

# PART V — PARALLEL EXECUTION AND DEPENDENCIES

The following work should start concurrently.

| Workstream | SIGNAL | FOODMECHANICS |
|---|---|---|
| Source discovery | S1 | F1 |
| Schema design | S2 | F2 |
| App implementation | S3 | F3 |
| Genie design | S4 | F4 |

The projects must remain independently runnable.

## Phase 1 — Immediate work, no Databricks required

Start all eight agents concurrently.

S1 and F1:
- Source research.
- API access.
- Initial real data.
- Fixture generation.

S2 and F2:
- Schema definitions.
- Local transformation pipelines.
- Data validation.

S3 and F3:
- Application shell.
- Navigation.
- Visualizations.
- Local data adapters.

S4 and F4:
- Genie instruction drafts.
- SQL examples.
- Evaluation questions.
- Expected-answer definitions.

The agents should not wait for another agent's entire implementation.

Instead, the orchestrator establishes provisional contracts early and lets independent development proceed.

## Phase 2 — Local end-to-end applications

Before Databricks access:

Both projects must demonstrate:

1. Source data ingestion.
2. Raw record preservation.
3. Normalization.
4. Deduplication.
5. Analytical queries.
6. Interactive application.
7. Source evidence.
8. Simulated incremental updates.
9. Evaluation tests.

Use local Parquet/JSON and DuckDB or another lightweight local analytical engine where appropriate.

Implement a storage and query adapter so the application can later switch to Databricks SQL without rewriting its business logic.

Do not make the application depend on mocked Genie answers.

If Genie is unavailable, show a clear unavailable state and retain normal structured-data exploration.

## Phase 3 — Databricks integration

When workspace access becomes available:

1. Create or identify the Unity Catalog catalog and schemas.
2. Establish raw landing locations.
3. Connect incremental ingestion.
4. Build Delta Bronze tables.
5. Build normalized Silver tables.
6. Build analytical Gold tables and views.
7. Validate row counts and data quality.
8. Configure the two separate Genie Agents.
9. Add the appropriate tables and views to each Agent.
10. Add instructions, semantic definitions, and example SQL.
11. Run the evaluation question sets.
12. Connect the custom applications to their respective Genie Agents.

Do not create unnecessary continuous compute just to demonstrate streaming.

If Auto Loader can incrementally process arriving files with a triggered pipeline, demonstrate that. If a source genuinely emits frequent events and the use case benefits from low latency, demonstrate streaming separately.

## Phase 4 — Final integration and demonstration

Each project must have its own complete demonstration.

SIGNAL:
- Introduce a new AI development.
- Show ingestion.
- Show entity extraction and relationships.
- Show the ecosystem update.
- Investigate the change with Genie.

FOODMECHANICS:
- Introduce a new ingredient or scientific evidence record.
- Show ingestion.
- Show nutritional or functional-property normalization.
- Show the knowledge relationships update.
- Investigate a food science question with Genie.

Both applications should have a reliable fallback demonstration using their fixture datasets.

---

# PART VI — SHARED INTEGRATION CONTRACTS

The orchestrator must establish these contracts before agents begin substantial implementation.

## 1. Application data interface

Each app should use a common conceptual interface:

```python
class DataProvider:
    def get_recent_records(self, limit=20):
        raise NotImplementedError

    def search(self, query, filters=None):
        raise NotImplementedError

    def get_detail(self, record_id):
        raise NotImplementedError

    def get_relationships(self, entity_id):
        raise NotImplementedError

    def get_metrics(self, filters=None):
        raise NotImplementedError
```

Implement separate providers:

- LocalDataProvider
- DatabricksDataProvider

The application should not care which provider supplies the data.

## 2. Genie interface

Create a separate integration layer for each application.

Conceptually:

```python
class GenieProvider:
    def ask(self, question, context=None):
        raise NotImplementedError
```

The real implementation should connect to the relevant Databricks Genie Agent using supported authentication and APIs.

The local implementation must not pretend to be Databricks Genie.

## 3. Source evidence

Every important extracted claim or relationship should retain:

- Source identifier.
- Source URL.
- Publication or release timestamp.
- Ingestion timestamp.
- Extraction timestamp.
- Confidence or evidence classification, where appropriate.
- Original source reference.

## 4. Evaluation

Each application must have a small set of deterministic test cases.

SIGNAL:
- Does a model comparison use the correct versions?
- Are benchmark methodologies distinguished?
- Does a development link to the correct source?
- Are release dates and ingestion dates distinct?
- Does a new record update the right analytical view?

FOODMECHANICS:
- Are nutrition values normalized correctly?
- Are serving bases preserved?
- Does an ingredient connect to the right functional property?
- Does a mechanism retain its supporting evidence?
- Are hypotheses distinguished from established findings?

---

# PART VII — COMPLETION CRITERIA

Do not consider a project complete just because it has a polished UI.

Each project must independently satisfy the following:

- [ ] A clear problem statement.
- [ ] At least two functioning data sources or defensible source-backed datasets.
- [ ] Raw data preservation.
- [ ] Structured normalized data.
- [ ] Incremental ingestion demonstration.
- [ ] Deduplication and data validation.
- [ ] Historical information or change tracking.
- [ ] Useful analytical views.
- [ ] A working interactive application.
- [ ] Source attribution.
- [ ] A dedicated Genie Agent configuration.
- [ ] A useful natural-language investigation experience.
- [ ] A tested evaluation set.
- [ ] A complete demonstration script.
- [ ] A local fallback if Databricks is unavailable.

The projects should be assessed independently. One project reaching completion must not conceal missing functionality in the other.

---

# PART VIII — FIRST INSTRUCTIONS TO ANTIGRAVITY

Execute the following in order:

1. Inspect the existing workspace and repository.
2. Preserve any existing work; do not overwrite or delete existing project files.
3. Create the two project directories and coordination directory.
4. Create the master progress tracker.
5. Create the shared data and application integration contracts.
6. Spawn eight independent implementation agents, four per project.
7. Start all independent research, schema, application, and Genie planning tasks concurrently.
8. Have the orchestrator review the initial schemas and interfaces before major implementation.
9. Require each agent to commit or clearly separate its deliverables.
10. Begin local end-to-end implementation immediately.
11. Do not wait for Databricks credentials or workspace setup.
12. When Databricks becomes available, integrate each project independently.

### Important agent management rules

- Do not allow agents to make competing changes to the same files.
- Assign file ownership.
- Use separate branches or isolated worktrees if supported.
- Keep task outputs small and verifiable.
- Require tests before accepting implementation.
- Avoid building speculative features before the main demo works.
- Do not let research agents spend the entire project collecting data.
- Do not let frontend agents build static mockups that cannot connect to real data.
- Do not let Genie agents write untested SQL examples.
- Do not let one project's schemas or assumptions leak into the other.

### Initial priority

The first milestone is NOT a complete Databricks deployment.

It is two independent, locally runnable applications that each demonstrate their core product idea using real or responsibly constructed data.

Databricks integration should strengthen the applications, not be a prerequisite for beginning them.

**Final objective:** Two polished, independently useful applications that demonstrate why Databricks is valuable for continuously ingesting, structuring, connecting, and investigating complex information.
