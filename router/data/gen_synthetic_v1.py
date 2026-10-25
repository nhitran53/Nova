"""FROZEN v1 generator: reproduces data/{train,dev,test}.jsonl (the data v1 was trained on) byte-for-byte.

    python data/gen_synthetic_v1.py         # writes data/{train,dev,test}.jsonl next to this file

Do not edit: tests/test_data_coverage.py regenerates and compares. Known weakness (review 2026-10-24): template
*pools* are disjoint across splits, but slot families such as {take} expand to the same frames in train, dev and
test, and dev/test share their eval-only slot pools. The v2 generator (gen_synthetic.py) fixes that.

Deterministic synthetic router data: JSONL {"text", "label"} for every label in labels.json.

Design (v1, after the "take me home" / "take me to home" brittleness bug):
  * Many phrasings per class: several verbs/frames, politeness wrappers, word-order variants, with and without
    "to/the/my", contractions and speech-ish noise (fillers, dropped function words, STT-style slips, typos).
  * One shared PLACES list (home, homebase, home base, base, habitat, airlock, rover, ...) is substituted into
    navigation, map, safety, oxygen, chit-chat (NO_TOOL), and unrelated (OOD) templates, so a place word alone
    never decides the class.
  * call_llm covers general space/science/how-it-works questions; OOD is unrelated topics (food, sports, Earth
    weather, shopping, gibberish ...).
  * Each label has separate template pools for train / dev / test (disjoint, see tests/test_data_coverage.py).
  * Texts that equal (after normalisation) a line of demo_calls*.json or paraphrase_test.jsonl are never emitted,
    so those hand-written evaluation sets stay verbatim-held-out. They are read for exclusion only.

A template starting with "~" is an imperative that may get a politeness prefix ("could you ...").

HONESTY NOTE: this is placeholder synthetic data from a template family. Dev/test come from the same generator
family, so scores on them are optimistic and say nothing about real speech.
"""
import argparse
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).parent
LABELS_PATH = HERE.parent / "labels.json"
SPLIT_SIZES = {"train": 200, "dev": 40, "test": 40}
HELD_OUT_FILES = (HERE.parent / "demo_calls.json", HERE.parent / "demo_calls_47.json", HERE / "paraphrase_test.jsonl")

# ---- shared place list: noun -> determiners it takes ("" = bare, e.g. "to home"). ----
# place words appear in every kind of template, so they never decide the class on their own.
PLACES = {
    "home": ["", "", "my", "our"],
    "homebase": ["", "the", "our"],
    "home base": ["", "the", "our"],
    "base": ["the", "the", "our", ""],
    "base camp": ["", "the", "our"],
    "habitat": ["the", "the", "our", "my"],
    "hab": ["the"],
    "airlock": ["the", "the", "our"],
    "rover": ["the", "the", "my", "our"],
    "lander": ["the", "our"],
    "crater": ["the", "that"],
    "crater rim": ["the"],
    "ridge": ["the", "that"],
    "landing pad": ["the", "our"],
    "landing site": ["the", "our"],
    "outpost": ["the", "our"],
    "greenhouse": ["the", "our"],
    "supply depot": ["the"],
    "comms tower": ["the"],
}
EVAL_ONLY_PLACES = {"relay tower": ["the"], "solar farm": ["the", "our"], "sample depot": ["the"], "dune field": ["the"]}
TRAIN_ONLY_PLACES = {"ice shelf": ["the"], "old quarry": ["the"], "lava tube": ["the"], "east outcrop": ["the"]}
HOME_NOUNS = ("home", "homebase", "home base", "base", "base camp")
BARE_HOME_NOUNS = ("home", "homebase", "home base")  # "take me home" has no "to"
DEST_FORMS = ["to {P}", "to {P}", "to {P}", "back to {P}", "back to {P}", "over to {P}", "toward {P}", "back toward {P}"]

# ---- slot pools: (shared, train-only, eval-only). dev/test use shared + eval-only. ----
POOLS = {
    "dir": (["north", "south", "east", "west"], ["northeast", "southwest"], ["northwest", "southeast"]),
    "hazard": (
        ["a dust storm", "a rockslide", "a radiation spike", "a suit leak"],
        ["thin ice", "low pressure", "bad visibility"],
        ["a solar flare", "unstable ground", "a thermal drop"],
    ),
    "sensor": (
        ["suit pressure", "battery", "heart rate", "core temperature", "CO2 level"],
        ["radiation dose", "wheel torque", "water supply", "blood pressure"],
        ["signal strength", "suit temperature", "pulse", "coolant level", "body temperature"],
    ),
    "topic": (
        ["regolith", "radiation shielding", "solar flares", "the water recycler", "orbital mechanics", "neutron stars",
         "tides", "photosynthesis", "magnetic fields"],
        ["the suit scrubber", "lava tubes", "low gravity walking", "dust on solar panels", "time dilation",
         "pressure suits", "the big bang", "plate tectonics", "gravity assists", "dark matter"],
        ["ice mining", "cosmic rays", "thermal insulation", "habitat domes", "the speed of light", "supernovae",
         "ocean currents", "the greenhouse effect", "asteroid belts"],
    ),
    "device": (
        ["a solar panel", "a fuel cell", "a heat pump", "a water filter", "a radio"],
        ["a battery", "a telescope", "a thermometer", "a gyroscope", "a rocket engine"],
        ["a gps receiver", "a spectrometer", "a pressure valve", "an electrolyzer", "a radiator"],
    ),
    "planet": (
        ["mars", "the moon", "venus", "jupiter"],
        ["saturn", "mercury", "europa"],
        ["titan", "neptune", "pluto"],
    ),
    "equip": (["the airlock", "the rover", "the habitat", "the lander"], ["the rebreather"], ["the life support system"]),
    "o2": (["oxygen", "O2"], ["air supply", "air"], ["breathable air", "oxygen reserve", "breathing gas"]),
    "city": (["London", "Chicago", "Berlin"], ["Tokyo", "Denver", "Seattle"], ["Madrid", "Boston", "Sydney", "Toronto"]),
    "sport": (["football", "basketball"], ["baseball", "hockey", "tennis"], ["soccer", "golf", "cricket", "rugby"]),
    "food": (
        ["pasta", "tacos", "pizza"],
        ["pancakes", "chicken curry", "lasagna", "meatloaf"],
        ["risotto", "brownies", "an omelette", "fried rice", "banana bread"],
    ),
    "language": (["Spanish", "French"], ["German"], ["Italian", "Japanese", "Portuguese"]),
    "stuff": (
        ["a new phone", "running shoes", "a laptop"],
        ["a used car", "a winter coat", "headphones"],
        ["a coffee maker", "a bicycle", "a backpack"],
    ),
    "celeb": (["Taylor Swift", "Tom Hanks"], ["Beyonce", "Brad Pitt"], ["Keanu Reeves", "Adele"]),
    "greet": (["hello", "hi", "hey"], ["good morning", "good evening", "howdy"], ["hiya", "greetings", "morning", "yo"]),
    "thanks": (["thanks", "thank you"], ["thanks a bunch", "cheers", "thanks a lot"], ["much appreciated", "thank you so much"]),
    "ack": (["okay", "got it", "cool"], ["understood", "roger", "copy that", "alright"], ["sounds good", "makes sense", "fair enough"]),
    "nova": (["nova"], ["buddy", "friend"], ["partner", "assistant"]),
    "take": (
        ["take me", "bring me", "get me", "lead me", "guide me", "walk me"],
        ["escort me", "direct me", "steer me"],
        ["show me the way", "carry me", "help me get"],
    ),
    "goverb": (["go", "head", "walk", "travel"], ["move", "drive"], ["make my way", "trek"]),
    "navverb": (["navigate", "route me", "set a course"], ["plot a course", "start navigation"], ["begin guidance", "give me a route"]),
}
SYLLABLES = {
    "train": ["ba", "ko", "ri", "zu", "me", "fa", "do", "la", "ne", "shi", "tor", "vek"],
    "dev": ["gu", "pe", "xa", "ni", "bo", "tre", "qua", "sol", "dim", "fro"],
    "test": ["wi", "ja", "ke", "mu", "pra", "zon", "hep", "lo", "ci", "dra"],
}

# ---- template pools per label and split ----
TEMPLATES = {
    "call_safety": {
        "train": [
            "is it safe to {goverb} {dest}", "can I make it {dest} and back", "is it a go for {place}",
            "any hazards between here and {place}", "check if {hazard} is a risk right now",
            "do I have enough margin to reach {place}", "is it safe to stay out another {mins} minutes",
            "run a safety check", "go or no go for {place}", "am I clear to leave {place}",
            "should I abort the walk {dest}", "is the path {dest} safe",
            "what's the risk of {hazard} near {place}", "can I safely spend {mins} minutes at {place}",
            "safety check for a trip {dest}", "will I make it back from {place}",
            "is it dangerous to cross {place}", "give me a go no go", "check hazards",
            "how risky is it to {goverb} {dest} with {hazard}", "is it safe to {goverb} home right now",
            "is it okay to {goverb} {dest}", "are there any dangers on the route {dest}",
            "will I be okay if I {goverb} {dest}", "what's the danger level around {place}",
            "is {hazard} a problem for the trip {dest}", "~check if it's safe to leave {place}",
            "~scan for hazards near {place}", "can I get {dest} before {hazard} hits", "safe to {goverb} {dest}",
            "is it too risky to {goverb} {dest} now", "any risk walking {dest}", "do you see any hazards ahead",
            "~tell me if the way {dest} is safe", "am I in danger out here", "how safe is it out here",
            "is it still safe at {place}", "is it ok to leave {place} now", "any warnings for the trip {dest}",
            "should I wait out {hazard} before going {dest}",
        ],
        "dev": [
            "would it be safe to walk out {dest}", "tell me if I'm good to go {dest}",
            "is {hazard} something I should worry about at {place}", "~do a hazard scan",
            "can I get {dest} and still get home", "am I cleared for {place}",
            "is it okay to continue {dest}", "is the way {dest} clear", "is it risky to {goverb} {dest}",
            "~verify the route {dest} is safe", "any threat from {hazard} on the way {dest}",
            "do you think I should {goverb} {dest} right now", "is this a bad time to {goverb} {dest}",
            "how dangerous is the trip {dest}",
        ],
        "test": [
            "any danger ahead", "is going {dest} a bad idea", "safe or not to head {dest}",
            "evaluate the risk of a {mins} minute stay at {place}", "is it wise to keep going with {hazard}",
            "check whether I can return {dest} safely", "~verify it's safe at {place}",
            "will {hazard} be an issue {dest}", "is it smart to {goverb} {dest}", "what are the risks {dest}",
            "are conditions good enough to {goverb} {dest}", "~assess the danger around {place}",
            "can I safely make the trip {dest}", "am I going to be okay {dest}",
        ],
    },
    "call_navigation": {
        "train": [
            "how do I get {dest}", "which way is {place}", "~navigate {dest}", "~give me directions {dest}",
            "route {dest}", "what's my heading", "~head {dir} for {dist}", "what direction is {place}",
            "{take} {dest}", "{take} {dest}", "{take} {dest}", "{take} {dest} please", "~{take} {dest}",
            "which way should I go to reach {place}", "what bearing to {place}", "~plan a route {dest}",
            "~start navigation {dest}", "turn me toward {place}", "~point me {dest}", "~set course for {place}",
            "next turn please", "what's the best route around {place}", "turn {dir} at the next marker",
            "{take} home", "{take} home", "~{take} home please", "I want to {goverb} {dest}", "I need to get {dest}",
            "I wanna go {dest}", "let's {goverb} {dest}", "{goverb} {dest}", "~{goverb} {dest} now",
            "where is {place}", "where's {place} from here", "how far is {place}", "which way to {place}",
            "directions {dest}", "{place} directions please", "route to {place} please", "okay {take} {dest}",
            "how do I {goverb} {dest}", "what's the way {dest}", "~{navverb} {dest}", "which way do I walk {dest}",
            "I'm lost, {take} {dest}", "which way is home", "where's home", "how do I get home",
            "~find me a path {dest}", "show me how to get {dest}",
        ],
        "dev": [
            "walk me {dest}", "what compass heading gets me {dest}", "I need directions {dest}",
            "which direction do I face for {place}", "lead the way {dest}", "~re-route me around {place}",
            "how do I reach {place} from here", "could you {take} {dest}", "{take} back home", "would you {take} {dest}",
            "I should {goverb} {dest}", "which way am I going {dest}", "~{navverb} {dest} when ready", "I'd like to go {dest}",
            "what's the direction of {place}", "point the way home",
        ],
        "test": [
            "get me {dest}", "bearing to {place} please", "tell me where to turn", "what's the quickest path {dest}",
            "just {take} {dest}", "{take} home when you can", "steer me {dir} toward {place}", "am I heading the right way for {place}",
            "directions for {place}", "we should {goverb} {dest}", "can you {take} {dest}", "I have to get {dest}",
            "how can I reach {place}", "navigation {dest}", "it's time to head {dest}", "where is home from here",
        ],
    },
    "call_map": {
        "train": [
            "show the map", "~open the map", "zoom in on the map", "zoom out", "~zoom in on {place}",
            "center the map on {place}", "~drop a waypoint at {place}", "add a waypoint here",
            "~mark {place} on the map", "show me the waypoints", "delete the last waypoint", "pan the map {dir}",
            "switch to the terrain map", "show the satellite layer", "~pull up the map of {place}",
            "put a pin on {place}", "hide the map", "zoom to {dist} scale", "list my waypoints",
            "rename the waypoint to {place}", "show {place} on the map", "~open the map and zoom in on {place}",
            "map view please", "~set a waypoint for {place}", "where is {place} on the map",
            "can you show me {place} on the map", "~save {place} as a waypoint", "zoom in", "zoom out a little",
            "~bring up the map", "map zoom in on {place}", "pin {place}", "highlight {place} on the map",
            "I want to see {place} on the map", "~add {place} to my waypoints", "put a marker on {place}",
            "switch the map to {dir} up", "~zoom the map in on home", "remove the waypoint at {place}",
            "recenter on {place}",
        ],
        "dev": [
            "bring up the map view", "can you zoom in a bit", "set a marker at {place}", "recenter the map on me",
            "turn on the contour layer", "remove that waypoint", "display waypoints near {place}",
            "~mark {place}", "map {place}", "show {place} on screen", "let me see the map of {place}",
            "~drop a pin at {place}", "make the map bigger", "zoom the map on {place}",
        ],
        "test": [
            "map please", "zoom way out", "save this spot as a waypoint", "~show me {place} on the map",
            "scroll the map {dir}", "toggle the grid overlay", "clear all waypoints", "~place a marker on {place}",
            "map of {place}", "~focus the map on {place}", "can I see the map", "track {place} with a waypoint",
            "open waypoints", "zoom in near {place}",
        ],
    },
    "call_data": {
        "train": [
            "what's my {sensor}", "read me the {sensor}", "give me a telemetry readout", "show sensor data",
            "how's my {sensor}", "~read all sensors", "what's the {sensor} reading", "status report",
            "what are the current readings", "give me my vitals", "what's the outside temperature",
            "~report {sensor}", "read out the {sensor} and the {sensor2}", "what is the ambient radiation",
            "~list all vitals", "update me on telemetry", "what's the {sensor} trend",
            "how are the suit systems doing", "pull the latest sensor values", "how's my {sensor} doing",
            "what is my {sensor}", "tell me my {sensor}", "~check my {sensor}", "{sensor} please",
            "what are my vitals", "give me my numbers", "I want my {sensor} and {sensor2}", "status",
            "~show me the readings", "how are my vitals", "~give me the suit status", "what do the sensors say",
            "~read the {sensor}", "what's the {sensor} at", "any change in my {sensor}", "how's everything reading",
            "what are my readings", "show my {sensor}", "how is my {sensor} looking",
            "what's my {sensor} right now",
        ],
        "dev": [
            "tell me the {sensor}", "what does the {sensor} say", "give me the numbers on my suit",
            "~dump the telemetry", "any change in {sensor}", "how's the battery doing", "~report vitals",
            "what's the {sensor} look like", "~display my {sensor}", "suit readout please", "how are my numbers",
            "I'd like to know my {sensor}", "can you read me the {sensor}", "what are the vitals",
        ],
        "test": [
            "I need my {sensor} reading", "what are my current readings", "sensor check", "~read back the {sensor}",
            "current suit status", "how high is the {sensor}", "show me live telemetry numbers", "{sensor} reading",
            "~get me the {sensor}", "what's the latest on my {sensor}", "how's my body doing", "~check the readouts",
            "what's the {sensor} number", "give me a rundown of my sensors",
        ],
    },
    "call_o2": {
        "train": [
            "check my {o2} level", "how much {o2} do I have left", "what's my {o2}", "{o2} status",
            "how long will my {o2} last", "what's my {o2} consumption rate", "how fast am I using {o2}",
            "how much {o2} is in my tank", "{o2} remaining", "am I burning {o2} too fast",
            "read the {o2} tank pressure", "how many minutes of {o2} left", "what's the {o2} percentage",
            "~report {o2} reserves", "{o2} level please", "how's my {o2}", "what's the {o2} burn rate",
            "is my {o2} running low", "give me {o2} and consumption", "{o2} check",
            "will my {o2} last until I get {dest}", "do I have enough {o2} to get {dest}",
            "do I have enough {o2} for the trip {dest}", "how much {o2} will I need {dest}",
            "is my {o2} enough to {goverb} {dest}", "what's my {o2} like", "how's the {o2} holding up",
            "~tell me my {o2} level", "how much {o2} left in the tank", "{o2} usage", "~check the {o2}",
            "how long can I stay out on this {o2}", "how many minutes until I run out of {o2}",
            "will I run out of {o2} before I get {dest}", "what's my {o2} at",
            "am I low on {o2}", "{o2} left", "how much {o2} do I have", "how is my {o2}",
            "I'm worried about my {o2}, how much is left",
        ],
        "dev": [
            "how's my {o2} looking", "tell me the {o2} level", "what's the tank at for {o2}", "{o2} usage right now",
            "how long can I breathe on this tank", "remaining {o2} time", "show consumption of {o2}",
            "is there enough {o2} to get {dest}", "how's the {o2} going", "~give me the {o2} status",
            "how much {o2} is there", "can my {o2} get me {dest}", "what is the {o2} level",
            "is my {o2} okay",
        ],
        "test": [
            "do a quick {o2} check", "how much breathable air is left", "what's my {o2} draw",
            "{o2} reserve minutes", "is the {o2} supply holding", "what percent {o2} is left", "my {o2} usage rate",
            "will the {o2} last {dest}", "how much {o2} do I have to {goverb} {dest}", "~read me my {o2}",
            "what's left of my {o2}", "is my {o2} good", "how's the tank", "how much longer will my {o2} last",
        ],
    },
    "call_llm": {
        "train": [
            "explain how {topic} works", "why is {topic} important", "what's the difference between {topic} and {topic2}",
            "can you explain {topic}", "tell me about {topic}", "how does {topic} affect astronauts",
            "why does {topic} happen", "give me a quick explanation of {topic}", "what would happen if {topic} failed",
            "walk me through {topic}", "help me understand {topic}", "what should I know about {topic}",
            "describe {topic} in simple terms", "why do we need {topic}", "summarize what {topic} is",
            "what are the pros and cons of {topic}", "teach me about {topic}", "what causes {topic}",
            "I'm curious about {topic}, can you elaborate", "how would you deal with {topic}",
            "how does {device} work", "what is {device}", "how does {device} work in space",
            "why is {planet} so {adj}", "tell me about {planet}", "what's the atmosphere of {planet} like",
            "how cold does it get on {planet}", "how long is a day on {planet}", "could people live on {planet}",
            "how does {equip} work", "what is {equip} for", "why do we have {equip}",
            "what is {topic}", "what's {topic}", "why is space so cold", "how do astronauts sleep in orbit",
            "how big is the solar system", "why do stars twinkle", "how far away is the sun",
            "why is the sky blue", "how do planets form", "what makes the moon shine",
            "how old is the universe", "why does gravity pull things down", "what keeps satellites in orbit",
            "how do we know how far stars are", "what's inside a comet", "why do comets have tails",
            "what is a light year", "how do telescopes see so far",
        ],
        "dev": [
            "I'd like to understand {topic}", "what's going on with {topic}", "break down {topic} for me",
            "how come {topic} matters so much", "can you elaborate on {topic} versus {topic2}",
            "what do scientists think about {topic}", "give me the background on {topic}",
            "how is {device} built", "what's it like on {planet}", "why is {planet} {adj}",
            "what does {equip} do", "how do astronauts exercise in orbit", "why is the night sky dark",
            "what is dark energy", "how do galaxies form", "why do planets orbit the sun",
        ],
        "test": [
            "why is {topic} a thing", "explain {topic} like I'm new", "how is {topic} different from {topic2}",
            "what's the story with {topic}", "talk me through {topic}", "what are the tradeoffs of {topic}",
            "could you clarify {topic}", "what powers {device}", "can you describe {planet}",
            "what's the surface of {planet} like", "how does {equip} keep me alive",
            "why is it hard to land on {planet}", "how do satellites stay up", "what are asteroids made of",
        ],
    },
    "NO_TOOL": {
        "train": [
            "{greet}", "{greet} {nova}", "{thanks}", "{thanks} {nova}", "{ack}", "{ack} {thanks}",
            "how are you {nova}", "how are you doing today", "you're doing great", "good job {nova}", "nice work",
            "never mind", "forget it", "cancel that", "that's all for now", "talk to you later", "goodbye {nova}",
            "sorry about that", "I'm back", "just testing", "{greet} there {nova}", "{greet} {nova} I'm here",
            "I made it back to {place}", "I'm at {place} now", "we're at {place}", "I'm back at {place}",
            "I'm home", "home sweet home", "finally home", "we're home", "made it home", "I'm heading home",
            "{thanks} for getting me {dest}", "great, I'm at {place}", "{thanks}, I'm at {place} now",
            "good to be back at {place}", "I just got to {place}", "{greet}, I'm back home", "nice to meet you",
            "you're awesome", "okay {nova}", "that was great", "oh nice", "wow", "huh", "alright then",
            "I see", "interesting", "good night {nova}", "see you {nova}", "yes", "no", "yep", "nope",
            "perfect", "that sounds right", "I'll be right back", "hold on a second", "one moment",
        ],
        "dev": [
            "{greet} there", "{ack} {nova}", "you're a lifesaver", "see you later {nova}", "it's nice talking to you",
            "how's it going {nova}", "{thanks} for that", "I got to {place}", "I'm finally at {place}",
            "{thanks}, we're home", "good to be home", "oh that's cool", "right", "sure", "that works",
            "okay that makes sense", "I'm done for today",
        ],
        "test": [
            "{thanks} so much", "{greet} {nova} how are you", "that's helpful", "appreciate it", "all good now",
            "bye for now", "how have you been", "I've arrived at {place}", "{thanks}, back at {place}",
            "I'm home now", "{ack}, I'm home", "oh okay", "that's neat", "good stuff", "yeah that's fine",
            "much better",
        ],
    },
    "OOD": {
        "train": [
            "what's the weather in {city}", "will it rain in {city} tomorrow", "who won the {sport} game last night",
            "what's the score of the {sport} match", "how do I cook {food}", "give me a recipe for {food}",
            "how long do I boil {food}", "{gibberish}", "{gibberish} {gibberish}", "what's the temperature in {city} today",
            "how many calories in {food}", "translate hello into {language}", "what's the best way to learn {language}",
            "when does the {sport} season start", "my dog won't stop barking", "recommend a movie for tonight",
            "how do I fix a flat tire", "what time does the grocery store open", "{gibberish} {city}",
            "what's a good {food} place near me", "what should I cook at home tonight", "how do I fix the wifi at home",
            "what's on tv at home tonight", "how do I clean my kitchen", "what's the best place to buy {stuff}",
            "how much does {stuff} cost", "where can I get {stuff} near {city}", "who is {celeb} dating",
            "what's {celeb}'s latest movie", "tell me about {celeb}", "how do I change a lightbulb",
            "what's the capital of {city}", "is it snowing in {city}", "what's the traffic like in {city}",
            "how do I get to the airport in {city}", "what's the exchange rate for dollars to euros",
            "how do I write a resume", "who won the election", "what's the stock market doing",
            "play some music", "{gibberish} {gibberish} {gibberish}", "how do I lose weight", "set an alarm for seven",
            "what's the best {food} recipe", "how's the weather today", "will it be sunny this weekend",
            "who plays {sport} the best", "how many players are on a {sport} team", "my phone is broken",
            "how do I reset my email password",
        ],
        "dev": [
            "is it going to snow in {city}", "who's winning the {sport} championship",
            "what should I make for dinner with {food}", "{gibberish} {food} {gibberish}", "what's the humidity in {city} this week",
            "how do you say thank you in {language}", "where can I buy {food}", "what's a good book to read",
            "how much is {stuff}", "tell me something about {celeb}", "how do I fix my sink at home",
            "what's the news today", "will it be windy in {city}", "how do I make {food} at home",
            "my printer isn't working", "{gibberish} home",
        ],
        "test": [
            "do I need an umbrella in {city}", "who scored in the {sport} final", "how do I bake {food}",
            "{gibberish} {food}", "what's the forecast for {city}", "best movie this year",
            "how do I get a cheaper phone plan", "where should I buy {stuff}", "is {celeb} still famous",
            "what's the weather like at home", "how hot is it in {city}", "what's for lunch",
            "{gibberish} {city} {gibberish}", "tell me a good {sport} team",
        ],
    },
}
ADJ = ["red", "cold", "dusty", "bright", "hot", "small", "windy"]

# ---- noise ----
POLITE = {
    "train": ["please", "can you", "could you", "would you", "will you", "I need you to", "I want you to", "hey nova can you",
              "nova please", "go ahead and", "i'd like you to"],
    "eval": ["please", "can you", "could you", "would you mind", "I'd like you to", "nova can you", "just", "if you could"],
}
FILLERS = {
    "train": ["uh", "um", "hey nova", "okay nova", "so", "like", "nova", "hmm", "ok so", "yeah", "hey nova uh", "umm",
              "okay", "alright", "oh"],
    "eval": ["uh", "um", "hey nova", "nova", "er", "right so", "listen", "nova quick one", "well", "hold on", "okay so"],
}
SUFFIXES = {
    "train": ["please", "thanks", "now", "real quick", "you know", "nova", "for me"],
    "eval": ["please", "right now", "if you can", "thanks", "quick", "nova", "for me"],
}
DROPPABLE = {"the", "to", "my", "a", "is", "it", "me", "of", "and", "i", "do", "at", "for", "on", "back", "our"}
STT_SLIPS = {
    "train": {
        "o2": "oh two", "waypoint": "way point", "waypoints": "way points", "nova": "nover", "route": "root",
        "to": "two", "for": "four", "map": "mac", "heading": "hitting", "safe": "save", "regolith": "rego lith",
        "wheel": "will", "meters": "metres", "know": "no", "your": "you're", "homebase": "home base",
        "home base": "homebase", "base": "bass", "navigate": "navigator", "oxygen": "oxigen", "me": "my",
    },
    "eval": {
        "o2": "oh to", "waypoint": "way point", "waypoints": "way points", "nova": "no va", "route": "rout",
        "to": "too", "for": "fore", "oxygen": "oxigen", "heading": "heating", "telemetry": "telemetery",
        "hazards": "has ards", "zoom": "zoomed", "bearing": "baring", "there": "their", "rim": "rym",
        "homebase": "home bass", "home base": "homebase", "base": "bass", "airlock": "air lock", "habitat": "habitats",
        "take": "tick", "me": "my",
    },
}
CLEAN_RATE = 0.2


def _noun_pool(split):
    return {**PLACES, **(TRAIN_ONLY_PLACES if split == "train" else EVAL_ONLY_PLACES)}


class _Slots(dict):
    """Lazily picks a value per slot name; reuses it if the name appears twice in one template."""

    def __init__(self, rng, split):
        super().__init__()
        self.rng, self.split = rng, split

    def _place(self, key):
        """Returns (noun, text-with-determiner) for the current place slot."""
        pool = _noun_pool(self.split)
        nouns = list(pool)
        weights = [4 if n in HOME_NOUNS else 1 for n in nouns]  # home-ish places are the common case
        noun = self.rng.choices(nouns, weights)[0]
        det = self.rng.choice(pool[noun])
        return noun, (f"{det} {noun}" if det else noun)

    def __missing__(self, key):
        rng = self.rng
        base = key[:-1] if key not in POOLS and key.endswith("2") and key != "dest2" else key
        if key == "mins":
            value = str(rng.choice([5, 10, 15, 20, 30, 45]))
        elif key == "dist":
            value = f"{rng.choice([10, 20, 50, 100, 200, 500])} meters"
        elif key == "gibberish":
            syl = SYLLABLES[self.split]
            value = "".join(rng.choice(syl) for _ in range(rng.randint(2, 5)))
        elif key == "adj":
            value = rng.choice(ADJ)
        elif key == "place":
            value = self._place(key)[1]
        elif key == "dest":
            noun, text = self._place(key)
            if noun in BARE_HOME_NOUNS and rng.random() < 0.45:
                value = rng.choice([noun, noun, f"back {noun}"]) if noun == "home" else noun
            else:
                value = rng.choice(DEST_FORMS).format(P=text)
        else:
            shared, train_only, eval_only = POOLS[base]
            pool = shared + (train_only if self.split == "train" else eval_only)
            value = rng.choice(pool)
            if key != base:  # second slot of the same kind must differ from the first
                value = rng.choice([v for v in pool if v != self[base]])
        self[key] = value
        return value


def render(template, rng, split):
    return template.format_map(_Slots(rng, split))


def _clean(text):
    text = text[0].upper() + text[1:]
    question = re.match(r"(?i)(what|how|why|is|can|do|am|which|who|will|should|where|could|would|any)\b", text)
    return text + ("?" if question else ".")


def _typo(word, rng):
    """One STT/typing slip: drop, swap or double a letter."""
    i = rng.randrange(1, len(word) - 1)
    kind = rng.choice(["drop", "swap", "double"])
    if kind == "drop":
        return word[:i] + word[i + 1:]
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:] if i + 2 <= len(word) else word
    return word[:i] + word[i] + word[i:]


def add_noise(text, rng, split, imperative=False):
    """Clean (cased, punctuated) with p=CLEAN_RATE; otherwise speech-ish noise."""
    kind = "train" if split == "train" else "eval"
    if imperative and rng.random() < 0.3:
        text = rng.choice(POLITE[kind]) + " " + text
    if rng.random() < CLEAN_RATE:
        return _clean(text)
    text = re.sub(r"[^\w\s]", "", text.lower().replace("'", ""))
    words = text.split()
    slips = STT_SLIPS[kind]
    if rng.random() < 0.4:
        idx = [i for i, w in enumerate(words) if w in slips]
        for i in rng.sample(idx, min(len(idx), rng.choice([1, 2]))):
            words[i] = slips[words[i]]
    if rng.random() < 0.3 and len(words) >= 4:
        idx = [i for i, w in enumerate(words) if w in DROPPABLE]
        if idx:
            del words[rng.choice(idx)]
    if rng.random() < 0.08:
        idx = [i for i, w in enumerate(words) if len(w) >= 5]
        if idx:
            i = rng.choice(idx)
            words[i] = _typo(words[i], rng)
    if rng.random() < 0.1 and words:
        i = rng.randrange(len(words))
        words.insert(i, words[i])
    if rng.random() < 0.4:
        words = rng.choice(FILLERS[kind]).split() + words
    if rng.random() < 0.15:
        words += rng.choice(SUFFIXES[kind]).split()
    return " ".join(words)


def normalise(text):
    return " ".join(re.sub(r"[^\w\s]", "", text.lower().replace("'", "")).split())


def held_out_texts(paths=HELD_OUT_FILES):
    """Normalised texts of the hand-written evaluation sets (never emitted by the generator)."""
    texts = set()
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        if path.suffix == ".json":
            rows = json.loads(path.read_text())["calls"]
        else:
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        texts.update(normalise(r["text"]) for r in rows)
    return texts


def generate(labels, seed=0, sizes=SPLIT_SIZES, banned=None):
    """Returns {split: [{"text","label"}]}. Text is unique across all splits and labels."""
    merged_llm = "llm" in labels
    if merged_llm:
        labels = list(TEMPLATES)
    banned = held_out_texts() if banned is None else banned
    seen = {}
    out = {}
    for split in ("train", "dev", "test"):
        rows = []
        for label in labels:
            rng = random.Random(f"{seed}/{split}/{label}")
            templates = TEMPLATES[label][split]
            got, attempts = 0, 0
            while got < sizes[split]:
                attempts += 1
                if attempts > 200 * sizes[split]:
                    raise RuntimeError(f"cannot make {sizes[split]} unique {split} texts for {label}")
                template = templates[attempts % len(templates)]
                imperative = template.startswith("~")
                text = add_noise(render(template.lstrip("~"), rng, split), rng, split, imperative).strip()
                if text and text not in seen and normalise(text) not in banned:
                    seen[text] = label
                    rows.append({"text": text, "label": label if label.startswith("call_") and label != "call_llm" else "llm",
                                 "fine_label": label} if merged_llm else {"text": text, "label": label})
                    got += 1
        random.Random(f"{seed}/{split}/shuffle").shuffle(rows)
        out[split] = rows
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default=str(LABELS_PATH))
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    labels = json.loads(Path(args.labels).read_text())["labels"]
    missing = [label for label in labels if label not in TEMPLATES and label not in {"llm"}]
    if missing:
        raise SystemExit(f"no templates for labels: {missing}")
    data = generate(labels, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in data.items():
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        (out_dir / f"{split}.jsonl").write_text(text)
        print(f"{split}: {len(rows)} rows -> {out_dir / (split + '.jsonl')}")


if __name__ == "__main__":
    main()
