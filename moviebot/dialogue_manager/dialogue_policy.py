"""A rule-based policy developed as an initial step to generate action by the
agent based on the previous conversation and current dialogue."""


import ast
import random
from copy import deepcopy
from typing import Any, Dict, List

from moviebot.core.intents.agent_intents import AgentIntents
from moviebot.core.intents.user_intents import UserIntents
from moviebot.dialogue_manager.dialogue_act import DialogueAct
from moviebot.dialogue_manager.dialogue_context import DialogueContext
from moviebot.dialogue_manager.dialogue_state import DialogueState
from moviebot.nlu.annotation.item_constraint import ItemConstraint
from moviebot.nlu.annotation.operator import Operator
from moviebot.nlu.annotation.slots import Slots
from moviebot.ontology.ontology import Ontology


class DialoguePolicy:
    def __init__(self, ontology: Ontology, isBot: bool, new_user: bool) -> None:
        """Loads all necessary parameters for the policy.

        Args:
            ontology: Rules for the slots in the database.
            isBot: If the conversation is via bot or not.
            new_user: Whether the user is new or not.
        """
        self.ontology = ontology
        self.isBot = isBot
        self.new_user = new_user

    def next_action(
        self,
        dialogue_state: DialogueState,
        dialogue_context: DialogueContext = None,
        restart: bool = False,
    ) -> List[DialogueAct]:
        """Decides the next action to be taken by the agent based on the
        current state and context.

        Args:
            dialogue_state: Current dialogue state.
            dialogue_context: Context of the dialogue. Defaults to None.
            restart: Whether or not to restart the dialogue. Defaults to False.

        Returns:
            A list of dialogue acts.
        """
        agent_dacts = []
        slots = deepcopy(dialogue_state.agent_requestable)

        # Restart conversation
        if restart or (
            UserIntents.RESTART
            in [dact.intent for dact in dialogue_state.last_user_dacts]
        ):
            agent_dacts.append(DialogueAct(AgentIntents.RESTART, []))
            agent_dacts.append(
                DialogueAct(
                    AgentIntents.ELICIT,
                    [ItemConstraint(slots[0], Operator.EQ, "")],
                )
            )
            return agent_dacts

        # Start conversation
        if not restart and (
            not dialogue_state.last_user_dacts
            or not dialogue_state.last_agent_dacts
            or all(
                [
                    UserIntents.HI == dact.intent
                    for dact in dialogue_state.last_user_dacts
                ]
            )
        ):
            agent_dacts.append(
                DialogueAct(
                    AgentIntents.WELCOME,
                    [
                        ItemConstraint("new_user", Operator.EQ, self.new_user),
                        ItemConstraint("is_bot", Operator.EQ, self.isBot),
                    ],
                )
            )
            return agent_dacts

        if any(
            [
                user_da.intent == UserIntents.ACKNOWLEDGE
                or user_da.intent == UserIntents.UNK
                for user_da in dialogue_state.last_user_dacts
            ]
        ) and any(
            [
                AgentIntents.WELCOME == agent_da.intent
                for agent_da in dialogue_state.last_agent_dacts
            ]
        ):
            return [
                DialogueAct(
                    AgentIntents.ELICIT,
                    [ItemConstraint(slots[0], Operator.EQ, "")],
                )
            ]

        # Define possible actions based on dialogue state
        possible_actions_per_state = {
            "agent_req_filled": self.elicit,
            # "agent_can_lookup": self.recommend,
            "agent_made_partial_offer": self.choose_elicit_or_recommend,
            "agent_should_make_offer": self.recommend,
            "agent_made_offer": self.inform_or_recommend,
            "agent_offer_no_results": self.no_results,
            "at_terminal_state": self.finish,
        }

        for state_name in ast.literal_eval(dialogue_state._agent_offer_state()):
            if state_name in possible_actions_per_state:
                agent_dacts = possible_actions_per_state[state_name](
                    dialogue_state, slots
                )
                if agent_dacts:
                    return agent_dacts

        return [DialogueAct(AgentIntents.CANT_HELP)]

    def finish(
        self, dialogue_state: DialogueState, slots: List[str]
    ) -> List[DialogueAct]:
        """Action to finish the conversation.

        Args:
            dialogue_state: Dialogue state.
            slots: Slots.

        Returns:
            List of dialogue acts to finish the conversation.
        """
        return [DialogueAct(intent=AgentIntents.BYE)]

    def no_results(
        self, dialogue_state: DialogueState, slots: List[str]
    ) -> List[DialogueAct]:
        """Action to communication that no results were found.

        Args:
            dialogue_state: Dialogue state.
            slots: Slots.

        Returns:
            List of dialogue acts indicating that no results were found.
        """
        return [DialogueAct(intent=AgentIntents.NO_RESULTS)]

    def elicit(
        self, dialogue_state: DialogueState, slots: List[str]
    ) -> List[DialogueAct]:
        """Action to elicit information need.

        Args:
            dialogue_state: Dialogue state.
            slots: Slots.

        Returns:
            List of dialogue acts to elicit information need.
        """
        agent_dacts: List[DialogueAct] = []

        CIN_slots = [
            key
            for key in dialogue_state.frame_CIN.keys()
            if not dialogue_state.frame_CIN[key] and key != Slots.TITLE.value
        ]

        if len(CIN_slots) >= dialogue_state.slot_left_unasked:
            agent_dacts.append(
                DialogueAct(
                    AgentIntents.COUNT_RESULTS,
                    [
                        ItemConstraint(
                            "count",
                            Operator.EQ,
                            len(dialogue_state.database_result),
                        )
                    ],
                )
            )
            # adding another dialogue act of ELICIT
            if dialogue_state.agent_req_filled:
                random.shuffle(CIN_slots)
                agent_dacts.append(
                    DialogueAct(
                        AgentIntents.ELICIT,
                        [ItemConstraint(CIN_slots[0], Operator.EQ, "")],
                    )
                )
            else:
                agent_dact = DialogueAct(AgentIntents.ELICIT, [])
                random.shuffle(slots)
                for slot in slots:
                    if not dialogue_state.frame_CIN[slot]:
                        agent_dact.params.append(
                            ItemConstraint(slot, Operator.EQ, "")
                        )
                        break
                agent_dacts.append(deepcopy(agent_dact))

        return agent_dacts

    def recommend(
        self, dialogue_state: DialogueState, slots: List[str]
    ) -> List[DialogueAct]:
        """Action to make a recommendation.

        Args:
            dialogue_state: Dialogue state.
            slots: Slots.

        Returns:
            List of dialogue acts to make a recommendation.
        """
        if dialogue_state.agent_should_make_offer:
            item_in_focus = dialogue_state.item_in_focus
        else:
            item_in_focus = dialogue_state.database_result[0]

        agent_dact = DialogueAct(
            AgentIntents.RECOMMEND,
            [
                ItemConstraint(
                    Slots.TITLE.value,
                    Operator.EQ,
                    item_in_focus[Slots.TITLE.value],
                )
            ],
        )

        return [agent_dact]

    def inform_or_recommend(
        self, dialogue_state: DialogueState, slots: List[str]
    ) -> List[DialogueAct]:
        """Action to communication that no results were found.

        Args:
            dialogue_state: Dialogue state.
            slots: Slots.

        Returns:
            List of dialogue acts indicating that no results were found.
        """
        agent_dacts: List[DialogueAct] = []
        for user_dact in dialogue_state.last_user_dacts:
            if user_dact.intent == UserIntents.INQUIRE:
                agent_dact = DialogueAct(AgentIntents.INFORM)
                for param in user_dact.params:
                    if param.slot != Slots.MORE_INFO.value:
                        agent_dact.params.append(
                            ItemConstraint(
                                param.slot,
                                Operator.EQ,
                                dialogue_state.item_in_focus[param.slot],
                            )
                        )
                    else:
                        agent_dact.params.append(
                            ItemConstraint(
                                param.slot,
                                Operator.EQ,
                                dialogue_state.item_in_focus[Slots.TITLE.value],
                            )
                        )
                if len(agent_dact.params) == 0:
                    agent_dact.params.append(
                        ItemConstraint(
                            "deny",
                            Operator.EQ,
                            dialogue_state.item_in_focus[Slots.TITLE.value],
                        )
                    )
                agent_dacts.append(deepcopy(agent_dact))
            elif user_dact.intent == UserIntents.ACCEPT:
                agent_dacts.append(
                    DialogueAct(
                        AgentIntents.CONTINUE_RECOMMENDATION,
                        [
                            ItemConstraint(
                                Slots.TITLE.value,
                                Operator.EQ,
                                dialogue_state.item_in_focus[Slots.TITLE.value],
                            )
                        ],
                    )
                )
        return agent_dacts

    def choose_elicit_or_recommend(
        self, dialogue_state: DialogueState, slots: List[str]
    ) -> List[DialogueAct]:
        """Action choosing between the elicit or recommend action.

        Args:
            dialogue_state: Dialogue state.
            slots: Slots.

        Returns:
            List of dialogue acts to either elicit information need or make a
            recommendation.
        """
        return self.elicit(dialogue_state, slots) or self.recommend(
            dialogue_state, slots
        )

    def _generate_examples(
        self, database_result: List[Dict[str, Any]], slot: str
    ) -> str:
        """Generates a list of examples for specific slot.

        Args:
            database_result: The database results for a user information needs.
            slot: Slot to find examples for.

        Returns:
            A string with a list of examples for a slot.
        """
        examples = []
        for result in database_result:
            temp_result = [x.strip() for x in result[slot].split(",")]
            examples.extend(
                [f"'{x}'" for x in temp_result if x not in examples]
            )
            if len(set(examples)) > 20:
                break
        if examples:
            examples = list(set(examples))
            random.shuffle(examples)
            if len(examples) == 1:
                return examples[0]
            _sub_example = [x for x in examples if len(x.split()) == 2]
            if len(_sub_example) >= 2:
                return " or ".join(random.sample(_sub_example, 2))
            else:
                return " or ".join(random.sample(examples, 2))
