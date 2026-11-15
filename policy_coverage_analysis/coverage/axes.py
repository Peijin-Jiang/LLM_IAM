"""The five classification axes, grouped into four dimensions.

Each axis is an independent yes/no question.  A scenario may satisfy any
number of them.  ``glossary`` entries are short forms the model would otherwise
read as opaque identifiers; they are there so it can understand the text, not
so it can pattern-match on them.  The judgement is made from the whole
description.

Every key begins with ``stated_`` on purpose.  These axes record what a
description *says*, not what the underlying model does.  Every IAM represents
supply-side technology and every mitigation run carries a price internally, so
"does the scenario have one" would be true everywhere and measure nothing.

Three of the four dimensions are named for a DEPARTURE from the default rather
than a mention -- an alternative to SSP2, policy action beyond what is already
in force, a technology assumption the scenario sets rather than one the model
merely has. A pathway is chosen from a set, so it is "alternative"; policy and
technology are things a scenario adds, so they are "new". What every run of a
model shares is not a design choice and is not counted.

The Mitigation constraint axis requires an explicit NUMBER.  A target and a
carbon price are one axis: a "carbon price" with no figure behind it is usually
the shadow price of a budget constraint - the same fact as the target rather
than a second one.  Requiring a figure, the axis measures a stated quantity
instead of a mechanism every mitigation run has.

"Policy" here means CLIMATE policy.  An SSP narrative carries policy content of
its own -- governance quality, environmental regulation, land protection -- but
the SSPs are defined as reference pathways WITHOUT climate policy, which is why
the framework layers Shared Policy Assumptions on top of them.  The
socioeconomic axis reads the pathway; this one reads the climate policy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def key_for(label: str) -> str:
    """The column name an axis label produces.

    One name is written per axis -- the label -- and the key is derived from it,
    so a workbook heading and a CSV column can never drift apart.  The prefix
    already carries "stated", so the word is dropped from the body, and a
    parenthetical gloss is not part of the name.
    """
    body = re.sub(r"\([^)]*\)", " ", label)
    body = re.sub(r"\bstated\b", " ", body, flags=re.IGNORECASE)
    return "stated_" + re.sub(r"[^0-9a-z]+", "_", body.lower()).strip("_")


@dataclass(frozen=True)
class Axis:
    label: str
    question: str
    glossary: tuple[str, ...]
    dimension: str = ""

    @property
    def key(self) -> str:
        """The CSV column name.  Derived, never written by hand."""
        return key_for(self.label)


DIMENSIONS: tuple[str, ...] = (
    "Alternative socioeconomic pathway",
    "Mitigation constraint",
    "New climate policy",
    "New technology assumption",
)


ADJECTIVES = (
    "Decide by what the sentence is about, not by its adjectives. Limited, "
    "limiting, constrained, restricted and default qualify warming levels, carbon "
    "budgets, policies and socioeconomic settings all through these descriptions. "
    "The adjective is evidence only when the thing it qualifies is {subject}. "
)

POLICY_IMPLIES = (
    "Do not infer a setting the description does not state. A policy package -- "
    "nationally determined contributions, implemented national policies, a carbon "
    "price -- reaches the technologies through constraints the text never spells "
    "out, and whatever it must be doing to them inside the model is not an answer "
    "to this question. "
    "A policy that DOES state a setting states one: how the scenario delivers a "
    "number is not what this question asks. A purchase subsidy of EUR 5000 on "
    "vehicles sets a cost, a mandate requiring 20 per cent synthetic fuels sets a "
    "deployment level, and both count here exactly as a learning rate or a cost "
    "taken from another case would. Such a statement is also new climate policy; "
    "the two questions are answered independently. "
)

RULE = (
    "The subject of the statement must be a named technology. A sector is not a "
    "technology, and a sentence explaining how the model represents abatement in "
    "general is not a technology. "
    "The statement must also SET OR CHANGE something about that technology in this "
    "scenario -- a cost, a cap or floor on it, a potential, a deployment level, a "
    "date it enters or leaves, an exclusion, or a mandate to use it. "
    "A limit that follows from WHAT THE TECHNOLOGY IS is a property, not a "
    "setting. Where it can physically be applied, what it is made of, how the "
    "model represents it, and whether it is in the portfolio at all are true of "
    "the technology in any run that has it: enhanced weathering using basalt on "
    "croplands in warm and temperate humid climate zones, afforestation "
    "restricted to temperate and tropical zones, the full carbon dioxide removal "
    "portfolio being available. Those are false here. A limit the SCENARIO chose "
    "is a setting: a cap with a value, a cost taken from a different case, an "
    "exclusion after a date, a rate of progress faster or slower than another "
    "run. Look for a choice that could have been made otherwise, not a fact about "
    "the technology. "
    "A consequence the description MARKS AS ONE is not a setting. Where the text "
    "itself says a choice leads to, results in or produces something -- divergent "
    "policies raise costs and lead to a quicker phase-out of oil use -- the second "
    "half is what the scenario arrived at, and the setting is in the first half. "
    "Listing what the IAM model has is not setting anything. Available CDR options are "
    "afforestation, BECCS and DACCS, the model represents wind and solar, nuclear is "
    "included in the portfolio: those state the IAM model\'s standing capability, "
    "and are false here. What counts is a departure a reader "
    "could compare against another scenario -- bioenergy capped at 100 EJ/yr, no CCS "
    "after 2050, photovoltaic costs taken from the low-cost case, electrification "
    "accelerated. "
)


AXES: tuple[Axis, ...] = (
    Axis(
        "Quantified mitigation constraint stated",
        "Does the description put a NUMBER on how hard this scenario has to "
        "mitigate -- either the end emissions must reach, or the price they carry? "
        "Either figure answers this question; the description needs only one of "
        "them. "
        "A target counts when its figure is given: a temperature in degrees (1.5 C, "
        "2 C), a cumulative carbon or greenhouse-gas budget in GtCO2, a "
        "radiative-forcing level in W/m2, a numeric emissions cap, or the year by "
        "which net zero or carbon neutrality is reached, or a well-below-2C "
        "pathway. "
        "A price counts when its figure is given: a tax level or path (15 "
        "US$2010/tCO2 from 2030, rising 5 per cent a year), a price stated for a "
        "given year, or a numeric cap or floor on the price. "
        "A figure the scenario ARRIVES AT is not a constraint it is held to. "
        "This scenario finally end up at approximately 2.5 C warming, emissions ending up at "
        "some level: those report where the run lands, and are false here however "
        "precise the number. What counts is a figure the run is limited to, "
        "constrained by, aiming at, imposed with, or consistent with. The question is "
        "how hard this scenario HAS TO mitigate, not how it turned out. "
        "Without a figure the answer is false, however clearly a constraint is "
        "named: limiting global warming, a stringent carbon budget, reaching net "
        "zero at some point, carbon prices grow and converge, a uniform global "
        "carbon price, prices rise steeply. Those name "
        "the mechanism every mitigation run has; the figure is what tells one "
        "scenario from another.",
        (
            "PkBudg500 / Budg600 / CB400 / _1000 -- a budget figure in GtCO2",
            "1p5 / 1.5C / 2C / 2Deg -- a temperature figure",
            "19I / 26I / 2.6 W/m2 -- a radiative-forcing figure",
            "net zero by 2050, carbon neutrality by 2060 -- a dated net-zero target "
            "the run is held to, not the year it happens to get there",
            "a carbon tax of 15 US$2010/tCO2 rising 5 per cent a year -- a price "
            "figure",
            "counter-example, answer false: limiting global warming, a stringent "
            "budget, a uniform global carbon price -- named, but no figure",
            "counter-example, answer false: it leads to approximately 2.5 C warming / "
            "warming exceeding 3 C -- the warming the run lands on, not a level it is "
            "held to",
            "counter-example, answer false: an NDC scenario whose emission constraints "
            "are based on quantifiable country targets -- quantifiable says a figure "
            "exists somewhere, it does not give one; the NDC itself is new climate "
            "policy and belongs to that question",
        ),
        dimension="Mitigation constraint",
    ),
    Axis(
        "Non-SSP2 future socioeconomic driver stated",
        "Does the description state a pathway other than SSP2 as the scenario's future "
        "socioeconomic driver? The driver means the long-run trajectory of population, "
        "GDP or income growth, demand drivers, and the development narrative that shapes "
        "them over the century. Answer true only when a different pathway shapes that "
        "long-run trajectory. If another SSP is invoked only to "
        "supply an assumption for a particular technology, such as its cost trajectory, "
        "that belongs to the technology axes and this axis is false. "
        "A near-term adjustment laid over an otherwise unchanged future is not a "
        "different driver: a COVID-19 sensitivity variant with a lower near-term gross "
        "domestic product trajectory still runs the future pathway it says it follows. "
        "Where the description states which pathway the scenario follows, that statement "
        "settles this question.",
        (
            "SSP1 (Sustainability), SSP3 (Regional Rivalry), SSP4, SSP5 (Fossil-fueled)",
            "an alternative population, GDP growth, income, or convergence pathway used as "
            "the driver -- not a near-term adjustment to one",
            "counter-example, answer false: a COVID-19 variant with a lower near-term "
            "GDP trajectory that still follows Middle of the Road (SSP2)",
            "SSP2 or Middle of the Road as the driver means false",
        ),
        dimension="Alternative socioeconomic pathway",
    ),
    Axis(
        "Supply-side technology setting stated",
        "Does the description give a named supply-side technology a setting of its "
        "own -- its cost, price, potential, deployment level, availability date, or "
        "any quantity specific to it? Answer true whenever a supply-side technology is "
        "the subject of such a statement, in whichever direction: a restriction, an "
        "expansion, a stated cost, or a stated amount all count equally. Supply "
        "side means energy conversion and carbon removal: renewables, solar, wind, "
        "nuclear, fossil fuels, coal, gas, CCS, BECCS, DACCS, "
        "hydrogen supply, bioenergy, and the grid, transmission, interconnection and "
        "storage that carry them. "
        + ADJECTIVES.format(subject="a supply-side technology") +
        "The standard narrative of a socioeconomic pathway is not a supply-side setting: "
        "SSP5 being described as relying more on carbon capture and storage, or SSP1 as "
        "having high renewable shares, characterises the pathway itself and belongs to "
        "the socioeconomic question, so it is false here unless the scenario states its "
        "own supply-side assumption on top of that narrative. "
        "An energy carrier -- bioenergy and biofuel, hydrogen, synthetic fuel -- sits on "
        "both sides. It belongs here only when the statement is about its production, "
        "supply volume, or cost. When the "
        "statement is instead about how much of it end users take up, that is a "
        "demand-side setting and this axis is false. " + RULE + POLICY_IMPLIES,
        (
            "lowBECCS / noBECCS / minCDR / noCCS / limCCS / NoNuc / lowNuc",
            "a stated technology cost, learning rate, potential, or deployment ceiling",
            "counter-example, answer false: a cap on the greenhouse-gas price effect in "
            "the agricultural sector -- that caps the price instrument in a sector, "
            "not a technology",
            "counter-example, answer false: marginal abatement cost data letting emissions "
            "respond to prices through control measures chosen by technical feasibility -- "
            "that describes how the model represents abatement, and names no technology",
            "counter-example, answer false: biofuel adoption is limited, in a "
            "scenario about vehicles -- that is end-user uptake of a carrier, so it "
            "belongs to the demand-side question instead",
            "counter-example, answer false: stylized technology policies and targets "
            "for a few major regions and countries -- technology qualifies the "
            "policies, the subject is regions, and no technology is given a setting",
            "counter-example, answer false: charging-station installation at 1 station "
            "per 2000 vehicles -- refuelling and charging infrastructure counted per "
            "vehicle or per user is an end-use adoption barrier, so it belongs to the "
            "demand-side question",
            "counter-example, answer false: divergent policies raise costs and lead to "
            "a quicker phase-out of oil use -- the text marks the phase-out as a "
            "consequence, so the setting is the divergence, not the oil",
        ),
        dimension="New technology assumption",
    ),
    Axis(
        "Demand-side technology setting stated",
        "Does the description give a named demand-side technology or service demand a "
        "setting of its own -- its cost, efficiency, uptake, demand level, a date it "
        "enters or leaves, or any quantity specific to it? Answer true whenever an end-use "
        "technology or demand driver is the subject of such a statement, in whichever "
        "direction. Demand side means end use: energy efficiency, buildings, appliances, "
        "heating and cooling, transport, modal choice, industry, electrification of end "
        "use, the uptake of biofuel, hydrogen or synthetic fuel by end users, behaviour, "
        "lifestyle, service demand levels, discount rates applied to end-use "
        "investment, and the refuelling or charging infrastructure end users need -- "
        "charging stations counted per vehicle, hydrogen refuelling counted per user "
        "-- which is an adoption barrier rather than part of the grid. "
        + ADJECTIVES.format(subject="a demand-side technology or a service demand") + RULE +
        "Two further exclusions. The standard narrative of a socioeconomic pathway is "
        "not a demand-side setting: SSP1 being described as involving sustainable "
        "lifestyles, high energy efficiency or high renewable shares characterises the "
        "pathway itself and belongs to the socioeconomic question, so it is false here "
        "unless the scenario states its own end-use assumption on top of that narrative. "
        "And a discount rate counts only when it applies to end-use investment -- "
        "buildings, appliances, vehicles. A global, social, or intertemporal discount "
        "rate governing the whole optimisation is a valuation parameter, not a "
        "demand-side technology, and is false here. "
        "An energy carrier -- biofuel, hydrogen, synthetic fuel -- belongs here when the "
        "statement is about its uptake by end users such as vehicles, buildings or "
        "industry. When the statement is instead about producing or supplying it, that is "
        "a supply-side setting and this axis is false. " + POLICY_IMPLIES,
        (
            "LowEnergyDemand / LED / EG (efficiency gap) / effLim",
            "implicit discount rates for buildings, appliances, heating or vehicles -- "
            "these are end-use adoption barriers and do count",
            "a stated efficiency level, service-demand level, or modal shift",
            "biofuel, hydrogen or synthetic-fuel adoption by vehicles, buildings or "
            "industry -- end-user uptake of a carrier counts here",
            "charging-station installation at 1 station per 2000 vehicles rather than "
            "1 per 1000, accelerated charging-infrastructure deployment -- end-use "
            "adoption infrastructure, counted per vehicle",
            "low demand response, flexible demand -- how much end users shift their "
            "load is a demand-side setting, and the level is stated",
            "counter-example, answer false: a cap on the greenhouse-gas price effect "
            "in agriculture -- that caps the price instrument in a sector, not a "
            "technology",
            "counter-example, answer false: abatement responding to prices through control "
            "measures, with a phased-in maximum reduction potential, "
            "describes how the model represents abatement in general, and names no "
            "end-use technology",
            "counter-example, answer false: DR2p / a global discount rate averaging 2% -- "
            "a valuation parameter, not an end-use assumption",
        ),
        dimension="New technology assumption",
    ),
    Axis(
        "Side-unspecified technology setting stated",
        "Does the description set a technology parameter -- a cost, a learning rate, a "
        "rate of technological progress, a potential, a deployment level, an "
        "availability date -- without the text telling you whether it is supply-side "
        "or demand-side? "
        "This question is the residual of the other two. Where the text names a "
        "supply-side technology or an end-use one, answer there and false here; this "
        "axis exists only for a setting whose side genuinely cannot be read off the "
        "text. Typical shapes: a technology-cost sensitivity varying techno-economic "
        "parameters across low, medium and high levels; updated technology-cost "
        "assumptions; technological change stated as fast or slow for the scenario as "
        "a whole; a learning or innovation rate applied across the portfolio. "
        "A statement that names technology but fixes "
        "no level is false here as it is on the other two: that the model incorporates "
        "technological change. "
        + POLICY_IMPLIES,
        (
            "a technology-cost sensitivity varying techno-economic parameters across "
            "low, medium and high levels",
            "technological change is fast / slow, said of the scenario as a whole",
            "updated technology-cost assumptions are used",
            "counter-example, answer false: photovoltaic costs taken from the low-cost "
            "case, bioenergy capped at 100 EJ/yr -- a named supply-side technology, so "
            "the supply-side question takes it and this one is false",
            "counter-example, answer false: implicit discount rates for buildings and "
            "appliances -- a named end use, so the demand-side question takes it",
            "counter-example, answer false: the marginal abatement cost curves "
            "incorporate technological change -- a capability every run has, fixing no "
            "level",
            "counter-example, answer false: stylized technology policies and targets "
            "for a few major regions and countries -- technology qualifies the "
            "policies, and no technology parameter is set",
            "counter-example, answer false: an mc specification, a def specification -- "
            "a token naming a specification without saying what it specifies sets "
            "nothing a reader could compare, whichever side it might belong to",
        ),
        dimension="New technology assumption",
    ),
    Axis(
        "New climate policy stated (beyond what is already in force)",
        "Does the description state that CLIMATE policy ACTION IS ADDED beyond what "
        "is already in force? "
        ""
        "WHAT COUNTS. Three kinds. First, a pledge or target not yet legislated and "
        "attributed to someone -- nationally determined contributions, a net-zero "
        "target countries submitted, a package a government announced or intends to "
        "adopt. Second, a concretely named new "
        "instrument the scenario switches on: a coal phase-out, an efficiency "
        "standard, a vehicle ban, a subsidy, good-practice policies rolled out, a "
        "sectoral regulation, international financial transfers. Third, a NAMED "
        "policy regime -- the Shared Policy Assumptions the SSP framework layers on a "
        "pathway: SPA0, SPA1, SPA2. "
        ""
        "WHAT DOES NOT COUNT. A continuation of what is already in force, however "
        "clearly it is named: current policies, implemented national policies, NPi, "
        "policies ratified by a cut-off date, business as usual. That is the starting "
        "point nearly every scenario in this database shares, and mentioning it "
        "reflects how much detail a description gives rather than what the scenario "
        "does. A scenario that adds nothing is false however specifically its "
        "emptiness is named: no additional policies, no future effects of policy "
        "changes beyond a given year, a business-as-usual reference pathway, ambition "
        "that does not increase, a baseline continuation of historical trends without "
        "climate or efficiency policies. "
        "Where a description states BOTH -- policies in force to some year, then new "
        "action after it -- the answer is true only when the new-action half names "
        "something. A handover from current policies to whatever meets the target is "
        "not an addition; it is where the constraint takes over. "
        ""
        "THE CARBON PRICE. A price that does no more than meet a target the "
        "description has already stated is that constraint's implementation and is "
        "not policy: a price introduced "
        "so the run reaches its budget, a price path timed to a policy start year, a "
        "price that rises and converges. A bare mitigation target is the same. But a "
        "scenario that DESIGNS SOMETHING AROUND the price states policy here, and the "
        "evidence is that design: international financial transfers, prices "
        "differentiated between regions by ability to pay or by an effort-sharing "
        "criterion, policy divergent across sectors and regions, or a named "
        "instrument switched on alongside the price. Ask what the description says "
        "BESIDES how high the price is and when it starts; if it says nothing "
        "besides, the answer is false. "
        ""
        "WHERE THE EVIDENCE MAY COME FROM. Three limits, all on the span itself. "
        "(a) It must come from the sentence saying what THIS RUN applies, allows or "
        "imposes. A description often opens by stating the question its STUDY explores "
        "-- a differentiated climate-policy analysis exploring the trade-off between "
        "cost efficiency and national sovereignty through international transfers -- "
        "and every member carries that "
        "sentence, its no-policy baseline included. That is the study's question, not "
        "this scenario's policy. "
        "(b) Where the study contains a member that ADDS NOTHING -- a current-policies "
        "run, a no-policy baseline -- that member is a worked answer, not a yardstick: "
        "whatever it states, it still adds nothing, so a sentence it also carries "
        "cannot be what makes a sibling an addition. NGFS represents climate policy as "
        "a carbon tax whose revenue is returned through lower income-tax rates in every "
        "one of its scenarios, its Current Policies run included, so the recycling is "
        "how that model works. Where a study has no such member this limit does not "
        "apply, and nothing follows from a sentence being shared. "
        "(c) Some descriptions do no more than restate the scenario name, saying that "
        "a scenario has a given specification. Judge such a description on what the "
        "token itself says. A token naming a pledge, a new programme or a policy-regime "
        "variant counts -- NDC, INDC, SPA1. A token naming the current-policy baseline "
        "does not, and neither does one that states no policy at all: a budget size, a "
        "temperature or forcing level, a dated net-zero target, or a transition "
        "speed.",
        (
            "NDC / INDC / INDCi / NDCp / nationally determined contribution / pledge",
            "SPA / Shared Policy Assumptions / SPA1 / SPA2 / SPA0",
            "a net-zero target ATTRIBUTED to someone -- announced, pledged, submitted, "
            "or held by named regions -- not one the run is simply held to",
            "good practice policies, GPP, a named regulation, standard, ban, mandate, "
            "phase-out, or subsidy newly introduced",
            "counter-example, answer false: NPi / NPi2020 / implemented national "
            "policies / current policies / policies ratified by 1 July 2019 -- the "
            "starting point, not added action",
            "counter-example, answer false: immediate introduction of comprehensive "
            "climate policies as of the 2020 policy start year -- the pricing that "
            "delivers the scenario's budget, named as policy but adding nothing",
            "international financial transfers, regional prices differentiated by "
            "ability to pay or by an effort-sharing criterion, policy divergent across "
            "sectors and regions -- policy designed around the price, and true",
            "counter-example, answer false: carbon-tax revenue returned through lower "
            "income-tax rates, where the study's current-policies run says it too -- "
            "the model's convention, not this scenario's addition",
            "counter-example, answer false: a differentiated climate-policy analysis "
            "exploring the trade-off between cost efficiency and national sovereignty "
            "-- the study's question, carried by every member including its baseline",
            "counter-example, answer false: Net-Zero 2050 / a 1.5 C pathway, with "
            "nothing else stated -- a target the run is held to, not a policy",
        ),
        dimension="New climate policy",
    ),
)

# The row for scenarios that stated nothing.  One name, like the axes.
OTHER = "None of the four stated"

DIMENSION_OF = {axis.label: axis.dimension for axis in AXES}

PROMPT_VERSION = "2026-08-27.7"

_TEMPLATE = """You are classifying integrated assessment model (IAM) \
scenarios from the AR6 database.

The {m} scenarios below are documented by the SAME paper or set of papers. They are \
one study, and its members differ only in the choices that study varied. Read all of \
them before answering any. The comparison is there to help you READ each \
description, and that is all it does:

- It tells you what a phrase is doing. A setting that separates two members is the \
thing that study varied, so a phrase carrying it is a design choice rather than \
background prose.
- It shows you which value is the DEFAULT. Where one member alters a quantity and \
another states that quantity unaltered, the study has just told you what its model \
does by default, and stating the unaltered value is not a setting. A study in which \
one scenario cuts buildings discount rates to 7 per cent while another reports them \
at baseline 10 to 20 per cent has a treatment in the first and the untouched \
default in the second: true there, false here, even though both carry a figure.
- It does NOT move the yardstick. Every question is asked against a fixed, external \
default -- SSP2, policies already in force, what the model can do in any run -- not \
against whatever this particular study happened to hold constant. A study whose \
members are ALL built on SSP1 has every member departing from SSP2, and the answer \
is true for all of them: no member alters the pathway, so nothing here shows SSP1 \
to be the default, and the default it departs from is external. Shared across the \
study never means "does not count" on its own.
- It never manufactures evidence. A scenario is judged on what ITS OWN description \
says; a sibling's wording is context for reading it, never a substitute for a span \
in it. A scenario that is itself the study's baseline is answered like any other: on \
what its own text states.

Read each description in full. It usually opens with the scenario name and then \
describes the experiment in prose. Both parts carry information and neither is \
sufficient on its own: the prose routinely states design choices the name does not \
encode, and the name sometimes encodes choices the prose never spells out.

Answer each of the {n} questions with true or false, separately for every scenario. \
The questions are independent; a scenario may satisfy any number of them. Each \
carries the design dimension it belongs to in brackets; two questions sharing a \
dimension are still separate questions and are answered separately.

Each question lists a few short forms you may encounter. That list is a glossary, \
not a checklist. A scenario can satisfy a question with none of those strings \
present, and a string being present does not settle the question when the \
surrounding text says otherwise.

For every answer you set to true you must also return FULL `evidence`: a span copied \
verbatim from THAT SCENARIO'S OWN description. Quote the phrase that actually \
decided it. Do not paraphrase, do not invent, do not quote a sibling. If you cannot \
find a verbatim span in that scenario's description, the answer is false. Leave \
`evidence` empty for every false answer.

Judge only what a description states. Do not infer from what a model of this type \
usually does, and do not treat the absence of a short form as evidence of absence.

Return one entry per scenario, all {m} of them, with `scenario` copied exactly as \
given.

{axes}

Scenarios:
{scenarios}"""


def build_prompt(items: list[tuple[str, str]]) -> str:
    """`items` is [(scenario, description), ...] - one study, or a slice of one."""
    axes = "\n\n".join(_axis_block(i, axis) for i, axis in enumerate(AXES, start=1))
    body = "\n\n".join(f"[{i}] {name}\n{text}"
                      for i, (name, text) in enumerate(items, start=1))
    return _TEMPLATE.format(m=len(items), n=len(AXES), axes=axes, scenarios=body)


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"scenario": {"type": "string"}} | {
                    axis.key: {
                        "type": "object",
                        "properties": {
                            "present": {"type": "boolean"},
                            "evidence": {"type": "string"},
                        },
                        "required": ["present", "evidence"],
                    }
                    for axis in AXES
                },
                "required": ["scenario"] + [axis.key for axis in AXES],
            },
        }
    },
    "required": ["results"],
}


def _axis_block(index: int, axis: Axis) -> str:
    glossary = "\n".join(f"    - {entry}" for entry in axis.glossary)
    return (f"{index}. {axis.key}  [{axis.dimension}]\n  {axis.question}\n"
            f"  Short forms you may encounter:\n{glossary}")


