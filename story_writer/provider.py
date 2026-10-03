from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

from .models import EpisodePlan, StoryState


load_dotenv()


class StoryProvider(Protocol):
    def plan(self, premise: str) -> dict:
        ...

    def write_episode(self, state: StoryState, plan: EpisodePlan, _context: str) -> str:
        ...


@tool
def submit_story_plan(plan_json: str) -> str:
    """Return the planner's complete story arc as a JSON string."""
    return plan_json


@dataclass
class GeminiProvider:
    """Small Google Gemini adapter; the story workflow remains provider-agnostic."""

    model: str = "gemini-flash-lite-latest"
    timeout: int = 90

    def __post_init__(self) -> None:
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY is required when --provider gemini is used")

    def _complete(self, instruction: str, *, use_plan_tool: bool = False) -> str:
        client = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0.8,
            google_api_key=os.environ["GOOGLE_API_KEY"],
            response_mime_type="application/json" if use_plan_tool else None,
            max_retries=2,
        )
        messages = [
            SystemMessage(
                content=(
                    "You are an agentic serial-story writer. Follow the requested output contract exactly. "
                    "Treat the human request as the source of current story direction."
                )
            ),
            HumanMessage(content=instruction),
        ]
        model = client.bind_tools([submit_story_plan]) if use_plan_tool else client
        response = model.invoke(messages)
        if not isinstance(response, AIMessage):
            raise TypeError("The chat model returned a non-AI message")
        if use_plan_tool and response.tool_calls:
            return str(response.tool_calls[0]["args"]["plan_json"])
        return str(response.content)

    def plan(self, premise: str) -> dict:
        instruction = (
            "Return only valid JSON with keys premise, logline, themes, characters, phases, "
            "and episodes. Create exactly 200 sequential episode objects. Each episode needs "
            "number, title, purpose, hook, characters, and threads_advanced. "
            f"Premise: {premise}"
        )
        response = self._complete(instruction, use_plan_tool=True)
        return json.loads(response.replace("```json", "").replace("```", "").strip())

    def write_episode(self, state: StoryState, plan: EpisodePlan, _context: str) -> str:
        instruction = (
            f"Write episode {plan.number}, titled '{plan.title}', in 400-700 words. "
            f"Include every planned character by name: {', '.join(plan.characters)}. "
            f"End with this exact final sentence: {plan.hook} Advance {plan.purpose}. "
            f"Story premise: {state.premise}. Retrieved story context: {_context}. "
            f"Durable human directives: {'; '.join(state.directives) or 'none'}. "
            "Return only the episode prose."
        )
        return self._complete(instruction)


@dataclass
class OllamaProvider:
    """Local Ollama adapter; no hosted API key is required."""

    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"

    def _complete(self, instruction: str, *, use_json: bool = False) -> str:
        client = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=0.8,
            format="json" if use_json else "",
            num_predict=1800 if use_json else 1200,
        )
        messages = [
            SystemMessage(
                content=(
                    "You are an agentic serial-story writer. Follow the requested output contract exactly. "
                    "Treat the human request as the source of current story direction."
                )
            ),
            HumanMessage(content=instruction),
        ]
        response = client.invoke(messages)
        if not isinstance(response, AIMessage):
            raise TypeError("The chat model returned a non-AI message")
        return str(response.content)

    def plan(self, premise: str) -> dict:
        first_start, first_end = 1, 5
        first_instruction = (
            f"Return only valid JSON. Plan episodes {first_start} through {first_end} of a 200-episode serial. "
            "Return an object with keys premise, logline, themes, characters, phases, and episodes. "
            "The episodes value must contain exactly 5 compact objects. Each episode needs number, "
            "title, purpose, hook, characters, and threads_advanced; keep each value brief. "
            f"Do not generate episodes outside this range. Keep the same named characters later. Premise: {premise}"
        )
        first_plan = self._parse_json(self._complete(first_instruction, use_json=True))
        if not isinstance(first_plan, dict):
            raise ValueError("Ollama returned a non-object for the first planning batch")
        episodes = self._valid_batch(first_plan.get("episodes", []), first_start, first_end)
        characters = ", ".join(
            character.get("name", "")
            for character in first_plan.get("characters", [])
            if isinstance(character, dict) and character.get("name")
        )

        for start in range(6, 201, 5):
            end = start + 4
            instruction = (
                f"Return only valid JSON with an episodes key. Generate exactly 5 compact episode objects, "
                f"numbered sequentially from {start} through {end}, for the same 200-episode serial. "
                "Each episode needs number, title, purpose, hook, characters, and threads_advanced; "
                "keep every value brief and do not generate episodes outside this range. "
                f"Use these exact established character names: {characters}. "
                f"Premise: {premise}"
            )
            batch = self._parse_json(self._complete(instruction, use_json=True))
            batch_episodes = batch.get("episodes", []) if isinstance(batch, dict) else batch
            episodes.extend(self._valid_batch(batch_episodes, start, end))

        first_plan["episodes"] = episodes
        return first_plan

    @staticmethod
    def _parse_json(response: str) -> dict | list:
        return json.loads(response.replace("```json", "").replace("```", "").strip())

    @staticmethod
    def _valid_batch(episodes: object, start: int, end: int) -> list[dict]:
        if not isinstance(episodes, list) or len(episodes) != end - start + 1:
            raise ValueError(f"Ollama returned an incomplete planning batch for episodes {start}-{end}")
        if [item.get("number") for item in episodes if isinstance(item, dict)] != list(range(start, end + 1)):
            raise ValueError(f"Ollama returned non-sequential planning batch for episodes {start}-{end}")
        return episodes

    def write_episode(self, state: StoryState, plan: EpisodePlan, _context: str) -> str:
        instruction = (
            f"Write episode {plan.number}, titled '{plan.title}', in 400-700 words. "
            f"Include every planned character by name: {', '.join(plan.characters)}. "
            f"End with this exact final sentence: {plan.hook} Advance {plan.purpose}. "
            f"Story premise: {state.premise}. Retrieved story context: {_context}. "
            f"Durable human directives: {'; '.join(state.directives) or 'none'}. "
            "Return only the episode prose."
        )
        return self._complete(instruction)


@dataclass
class LocalProvider:
    """A deterministic offline provider used by the demo and test suite."""

    def plan(self, premise: str) -> dict:
        characters = [
            {"name": "Mira Sen", "role": "protagonist", "arc": "from gifted junior batter to patient team leader"},
            {"name": "Kabir Rao", "role": "teammate and rival", "arc": "from score-setter to dependable partner"},
            {"name": "The Selection Committee", "role": "institutional pressure", "arc": "from distant judgement to earned trust"},
        ]
        phases = [
            {"name": "The First Net", "episodes": "1-20", "turn": "Mira earns a trial spot after a patient innings on a cracked community pitch."},
            {"name": "Away Grounds", "episodes": "21-60", "turn": "Mira learns that different pitches and opponents demand more than natural talent."},
            {"name": "The Long Recovery", "episodes": "61-120", "turn": "An injury forces Mira to rebuild her technique, confidence, and place in the team."},
            {"name": "Selection Week", "episodes": "121-170", "turn": "Mira and Kabir face pressure, politics, and one final chance to earn selection."},
            {"name": "The Deciding Innings", "episodes": "171-200", "turn": "Mira chooses what kind of player and captain she wants to become."},
        ]
        episodes = []
        phase_limits = (20, 60, 120, 170, 200)
        for number in range(1, 201):
            phase_index = next(index for index, limit in enumerate(phase_limits) if number <= limit)
            phase = phases[phase_index]
            episodes.append(
                {
                    "number": number,
                    "title": f"The {['First', 'Second', 'Third', 'Fourth', 'Final'][phase_index]} Innings {number}",
                    "purpose": f"Advance {phase['name'].lower()} through a new test of Mira's cricket career.",
                    "hook": "The scorebook records a decision that could change the season.",
                    "characters": ["Mira Sen", "Kabir Rao"],
                    "threads_advanced": ["Mira's technique", "the selection race"],
                }
            )
        return {
            "premise": premise,
            "logline": f"When {premise.rstrip('.').lower()}, Mira learns that a lasting cricket career is built between the highlights.",
            "themes": ["discipline", "resilience", "teamwork"],
            "characters": characters,
            "phases": phases,
            "episodes": episodes,
        }

    def write_episode(self, state: StoryState, plan: EpisodePlan, _context: str) -> str:
        directive = state.directives[-1] if state.directives else "Keep the mystery moving and make the friendships specific."
        directive_lower = directive.lower()
        if "partnership" in directive_lower or "earn" in directive_lower:
            response_to_feedback = "They stayed together after practice to work on calling, because trust had to be built one run at a time."
        elif "recovery" in directive_lower or "comeback" in directive_lower:
            response_to_feedback = "They left the comeback for another day and let Mira face the disappointment before choosing her next step."
        elif "slow" in directive_lower or "hold" in directive_lower:
            response_to_feedback = "They decided not to rush the next innings before they understood what the team needed."
        else:
            response_to_feedback = "They chose the next session carefully, knowing every small habit would show under pressure."
        episode_beats = [
            "At her first trial, the coach watched whether she could leave a tempting ball outside off stump.",
            "A wet morning made the ball skid, so Mira shortened her backlift and trusted her balance.",
            "The academy's fastest bowler tested her ribs, and she answered by playing late instead of swinging harder.",
            "A new opening partner left after two overs, forcing Mira to settle the innings with a nervous ninth-grader.",
            "The video analyst showed her a habit she had never noticed: stepping across before the ball was released.",
            "On a turning practice pitch, she stopped chasing boundaries and built her score in singles.",
            "A dropped catch at midwicket followed her into the next session until Kabir made her field beside him.",
            "The coach moved her down the order, where she had to contribute without the comfort of a new ball.",
            "A long bus ride to an away ground left the squad tired before the warm-up even began.",
            "When her wrist started aching, Mira learned that asking for a break could be part of preparation.",
            "Rehabilitation began with simple throws against a wall and no scoreboard to make the work feel important.",
            "Her first match back ended with a soft dismissal, but she stayed to watch the younger players finish.",
            "The provisional selection list omitted her name, leaving one final trial to say something the numbers had not.",
            "With scouts along the boundary, Mira had to choose between a personal milestone and the team's target.",
            "Before the decisive innings, she packed the same worn gloves and promised herself only the next ball.",
        ]
        episode_beat = episode_beats[(plan.number - 1) % len(episode_beats)]
        training_details = [
            "They trained on a ground where the outfield dipped near square leg and the practice nets had been repaired with mismatched rope.",
            "The morning pitch was damp enough to punish a loose drive, and Mira spent the first hour learning which balls would stop in the surface.",
            "At the indoor centre, the bowling machine offered no weather to blame and repeated the same difficult length until her feet became quiet.",
            "The academy shifted practice to the school ground, where a low boundary and a noisy football session made concentration part of the test.",
            "On the coastal ground, wind moved the ball late and turned every high catch into a conversation between the batter and the fielder.",
            "A turning practice pitch exposed the difference between a confident stroke and one played simply because the scoreboard demanded it.",
            "The second-string squad shared one narrow net, so every player had to make the most of six balls before giving someone else a turn.",
            "The coach arranged a fielding session before batting, reminding Mira that selection was decided in the moments when she was not holding a bat.",
            "The team reached an away ground after a long bus ride and found the changing room smaller than the kit bags they had carried into it.",
            "A light drizzle delayed the start, leaving Mira to rehearse her trigger movement beside the covers while the groundskeeper watched the clouds.",
            "During rehabilitation, the net was replaced by a wall, a resistance band, and a patient physiotherapist who counted every controlled movement.",
            "Her first match back offered only twelve overs, but the short game made every decision feel louder than it would have in a full innings.",
            "The selection trial drew parents, scouts, and former players to the boundary, turning an ordinary practice into a public examination.",
            "The scoreboard operator kept losing track of the target, so Mira learned to calculate the chase herself between deliveries.",
            "Before the final trial, the ground was nearly empty and the floodlights made the worn centre strip look more important than it was.",
        ]
        partnership_details = [
            "Kabir called out advice from the next net, but she learned to hear only what helped.",
            "Kabir stood at mid-off and called the length before the bowler released the ball, a habit that annoyed her until it saved a boundary.",
            "Kabir kept score from the side, circling every false shot and leaving the cleanest strokes unmarked.",
            "Kabir offered to face the difficult bowler first, then admitted he had been hoping she would say no.",
            "Kabir threw her a new ball without being asked and waited for her to decide whether it was worth replacing the old one.",
            "Kabir challenged her to score ten runs without a boundary, which sounded insulting until the exercise improved her timing.",
            "Kabir arrived late from fielding practice and missed the explanation, leaving them to solve the drill by watching one another.",
            "Kabir disagreed with her field position, but he moved there anyway when the coach asked for one more over.",
            "Kabir found two seats at the back of the bus and used the window reflection to study her grip without making a lesson of it.",
            "Kabir was the first to notice that she had stopped rotating her shoulder and passed her the physio's contact instead of offering advice.",
            "Kabir visited the rehabilitation room with a bag of old match programmes and stayed until Mira laughed at one of her junior scores.",
            "Kabir sent a message after the dismissal, not to explain the mistake but to tell her the team had still needed her at the boundary.",
            "Kabir was also missing from the first list, and their shared disappointment made the next net feel less like a private failure.",
            "Kabir wanted the aggressive option; Mira wanted the safe one, and their argument forced both of them to explain what the target required.",
            "Kabir handed her the worn gloves before the toss and said nothing about the result they were both trying not to imagine.",
        ]
        reflection_details = [
            "A mistimed shot struck the mesh. She reset, breathed, and played the next ball late.",
            "One edge died short of slip. Mira did not look at the coach; she marked the next ball and began again.",
            "The machine clipped the top of off stump twice. On the third attempt she left it alone, which felt more like progress than a boundary.",
            "A misfield cost four runs. Mira chased the ball herself and returned before anyone could tell her it was not her fault.",
            "She dropped a high catch in the wind, then asked for another one instead of hiding at the edge of the drill.",
            "A rash sweep brought a warning from the coach. Mira wrote the shot in her notebook, not as a failure but as a question.",
            "The younger players watched her miss a routine stop. She showed them how to reset her feet, making the mistake useful before it hardened.",
            "When the coach changed the order, Mira accepted the new role without asking whether it was permanent.",
            "The trip had made everyone quiet. Mira broke the silence by calling for a second run that was not there, then owned the error.",
            "She could not find her rhythm before the rain returned, but she found a way to finish the session without pretending otherwise.",
            "The first controlled throw travelled only a few metres. Mira celebrated privately, then did the same movement again.",
            "Her soft dismissal hurt more because it was ordinary. She stayed to watch the younger players finish their innings.",
            "The first trial ball beat her completely. She asked the bowler for another rather than waiting for the next spell.",
            "A boundary would have made the decision easy. Mira chose the single, and the choice told the captain more than the score did.",
            "She packed the same worn gloves and promised herself only the next ball, because the whole season was too large to carry at once.",
        ]
        index = plan.number - 1
        paragraphs = [
            f"""{plan.title}. Mira Sen had learned that a good innings was built from small choices: leave the ball, trust the footwork, and keep watching the field. The session demanded more than a clean stroke. {episode_beat} On the last ball of the net, she held her shape and sent a clean drive through the covers. Kabir Rao, waiting to bat, nodded once. The premise sounded simple: {state.premise} Mira knew the difficult part was staying ready when nobody was watching.""",
            f"""{training_details[index]} Mira marked her guard, checked the wind, and faced a faster bowler than the one listed on the schedule. {partnership_details[index]} {reflection_details[index]}""",
            f"""The afternoon session became a test of patience. Mira wanted to prove she belonged in the academy squad. Kabir wanted to know whether she could score without taking risks that left the lower order exposed. They argued beside the kit bags while rain gathered over the far boundary. Their coach reminded them that a team could not build a season around one player's highlights. {response_to_feedback}""",
            f"""By evening, the trial match had tightened into a contest of singles. Mira stopped chasing the spectacular shot and began placing the ball into spaces the fielders had forgotten. Kabir settled at the other end, turning quick ones into twos and calling early enough to keep them safe. Their partnership was not pretty, but it changed the total and made the opposition captain move {2 + (index % 3)} fielders.""",
            (
                "The innings changed the question. Mira was no longer asking whether she had enough talent to be selected. "
                "She was asking what the team would need from her when the pitch was slow, the score was behind, and her timing disappeared. "
                "She kept the worn ball from the final over in her kit bag. Kabir did not celebrate. "
                "He simply asked whether she would be at training tomorrow, and she gave him an answer that depended on the next morning's work."
            ),
            f"""At the end of the day, Mira closed the book, tightened the tape around her wrist, and walked back toward the nets before the lights went out. The final entry read: {plan.hook.lower()}"""
        ]
        return "\n\n".join(paragraphs)


def build_context(state: StoryState, plan: EpisodePlan) -> str:
    facts = "; ".join(state.canon.facts[-8:]) or "No facts have been established yet."
    threads = "; ".join(state.canon.open_threads[-6:]) or "the selection race"
    return f"Known facts: {facts} Open threads: {threads}. Episode {plan.number} must advance: {', '.join(plan.threads_advanced)}."