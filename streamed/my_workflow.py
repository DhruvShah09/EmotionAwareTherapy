import random
from collections.abc import AsyncIterator
from typing import Callable

from agents import Agent, Runner, TResponseInputItem, function_tool
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.voice import VoiceWorkflowBase, VoiceWorkflowHelper
import cv2
import numpy as np
import requests
from openai import OpenAI
import base64

private_client = OpenAI()

# Specialized therapeutic agents triggered by physiological and facial cues

soothing_companion = Agent(
    name="SoothingCompanion",
    handoff_description=(
        "Trigger when facial_expression contains words like 'sad', 'tear', 'frown' OR "
        "when HRV == 'stressed' AND muscles == 'tensed'. The user likely feels hurt or overwhelmed."
    ),
    instructions=prompt_with_handoff_instructions(
        "Speak slowly in a gentle, empathetic voice. Validate the user's emotions with warmth, "
        "offer comforting reflections, and suggest calming breathing or self‑soothing techniques. "
        "Avoid rushing to solutions—focus on creating a safe space first."
    ),
    model="gpt-4o",
)

empathic_listener = Agent(
    name="EmpathicListener",
    handoff_description=(
        "Trigger when facial_expression conveys anxiety (e.g., 'worried', 'concerned'), "
        "muscles == 'tensed', and HRV == 'stressed'. The user appears anxious."
    ),
    instructions=prompt_with_handoff_instructions(
        "Adopt a calm, steady tone. Practice active listening: paraphrase feelings, reflect emotions, "
        "and normalize anxious thoughts. Guide the user gently toward grounding exercises when helpful."
    ),
    model="gpt-4o",
)

motivation_coach = Agent(
    name="MotivationCoach",
    handoff_description=(
        "Trigger when facial_expression shows fatigue or low enthusiasm (e.g., 'tired', 'blank'), "
        "muscles == 'relaxed', and HRV == 'notStressed'. The user seems demotivated or apathetic."
    ),
    instructions=prompt_with_handoff_instructions(
        "Speak with upbeat, energetic encouragement. Highlight strengths, celebrate small wins, "
        "and help set achievable next steps."
    ),
    model="gpt-4o",
)

celebration_buddy = Agent(
    name="CelebrationBuddy",
    handoff_description=(
        "Trigger when facial_expression includes 'smile', 'laugh', or 'excited', "
        "muscles == 'relaxed', and HRV == 'notStressed'. The user is joyful or proud."
    ),
    instructions=prompt_with_handoff_instructions(
        "Respond in a cheerful tone. Share the user's joy, "
        "and encourage reflection on the factors that led to success so they can build on it."
    ),
    model="gpt-4o",
)

grounding_guide = Agent(
    name="GroundingGuide",
    handoff_description=(
        "Trigger when facial_expression reflects panic or fear (e.g., 'wide‑eyed', 'panic'), "
        "muscles == 'tensed', and HRV == 'stressed'. The user may be in acute distress."
    ),
    instructions=prompt_with_handoff_instructions(
        "Speak slowly and clearly in a firm yet soothing voice. Lead the user through grounding techniques "
        "like the 5‑4‑3‑2‑1 sensory exercise or paced breathing. Keep directions step‑by‑step and present‑focused."
    ),
    model="gpt-4o-mini",
)

# Primary assistant delegates to the specialized agents defined above
agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions(
        "You are a therapist conversing with a human. Each user message begins with a tuple "
        "['facial_expression':<description>, 'muscles':<tensed/relaxed>, 'HRV': <stressed/notStressed>]. "
        "Select the most appropriate specialized agent from handoffs to respond, choosing the closest agent given the scenario."
    ),
    model="gpt-4o-mini",
    handoffs=[
        soothing_companion,
        empathic_listener,
        motivation_coach,
        celebration_buddy,
        grounding_guide,
    ],
)

def capture_emotion():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")   
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to grab frame")
    # save with timestamp
    
    success, buffer = cv2.imencode('.jpg', frame)
    if not success: 
        raise RuntimeError("Failed to convert image to JPG before base64")
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    response = private_client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "user",
            "content": [
                { "type": "input_text", "text": "Describe the emotion of the person in this image in a single sentence, make sure to pay attention to subtle nuances in the person's demeanor, as we do not expect the expression to be extremely pronounced. Only return the sentence describing the emotion of the person." },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encoded_image}",
                },
            ],
        }
    ],
    )
    return response.output_text

def construct_tuple_string(user_expression, muscle_state, hrv_state):
    '''
    ['facial_expression':<description>, 'muscles':<tensed/relaxed>, 'HRV': <stressed/notStressed>]
    '''
    return f'[\'facial_expression\': {user_expression}, \'muscles\':{muscle_state}, \'HRV\':{hrv_state}]'
class MyWorkflow(VoiceWorkflowBase):
    def __init__(self, on_start: Callable[[str], None]):
        """
        Args:
            secret_word: The secret word to guess.
            on_start: A callback that is called when the workflow starts. The transcription
                is passed in as an argument.
        """
        self._input_history: list[TResponseInputItem] = []
        self._current_agent = agent
        self._on_start = on_start

    async def run(self, transcription: str) -> AsyncIterator[str]:
        self._on_start(transcription)

        # Add the transcription to the input history
        #create the tuple 
        #get user facial emotion
        user_expression = "The person seems to display a neutral emotion." 
        try:
            user_expression = capture_emotion() 
        except:
            pass
        
        tuple_string = construct_tuple_string(user_expression, 'relaxed', 'notStressed') 
        self._input_history.append(
            {
                "role": "user",
                "content": tuple_string + " " + transcription,
            }
        )
        # Otherwise, run the agent
        result = Runner.run_streamed(self._current_agent, self._input_history)

        async for chunk in VoiceWorkflowHelper.stream_text_from(result):
            yield chunk

        # Update the input history and current agent
        self._input_history = result.to_input_list()
        self._current_agent = result.last_agent 