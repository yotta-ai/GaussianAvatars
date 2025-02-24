from enum import Enum
from pydantic import BaseModel
from pydantic.types import List
from typing import Union


class Conversation(BaseModel):
    """
    Represents a conversation with the AI.
    """

    session_id: str
    items: List[
        Union[
            "SystemConversationItem",
            "UserConversationItem",
            "AssistantConversationMessageItem",
            "AssistantConversationFunctionCallItem",
            "AssistantConversationFunctionCallOutputItem",
        ]
    ] = []


class ConversationItemType(str, Enum):
    """
    Represents the types of conversation items.
    """

    message = "message"
    function_call = "function_call"
    function_call_output = "function_call_output"


class ConversationItemRole(str, Enum):
    """
    Represents the role of the conversation item
    """

    user = "user"
    system = "system"
    assistant = "assistant"


class ConversationItem(BaseModel):
    """
    Represents an item in the conversation.
    """

    id: str
    previous_id: str | None = None
    type: ConversationItemType
    role: ConversationItemRole
    # value: (
    #     "SystemConversationItem"
    #     | "UserConversationItem"
    #     | "AssistantConversationMessageItem"
    #     | "AssistantConversationFunctionCallItem"
    #     | "AssistantConversationFunctionCallOutputItem"
    #     | None
    # ) = None


class SystemConversationItem(ConversationItem):
    """
    Represents a system conversation item.
    """

    text: str | None = None


class UserConversationItem(ConversationItem):
    """
    Represents a user conversation item.
    """

    text: str | None = None
    audio: str | None = None


class AssistantConversationItem(ConversationItem):
    """
    Represents an assistant conversation item.
    """

    usage: Union["ConversationUsage", None] = None


class AssistantConversationMessageItem(AssistantConversationItem):
    """
    Represents an assistant conversation item.
    """

    text: str | None = None
    audio: str | None = None


class AssistantConversationFunctionCallItem(AssistantConversationItem):
    """
    Represents an assistant conversation function call item.
    """

    call_id: str
    name: str
    arguments: str|None = None


class AssistantConversationFunctionCallOutputItem(ConversationItem):
    """
    Represents an assistant conversation function call output item.
    """

    call_id: str
    output: str


class ConversationUsage(BaseModel):
    """
    Represents the usage of the conversation.
    """

    input_token_details: "ConversationInputTokenDetails"
    input_tokens: int
    output_token_details: "ConversationOutputTokenDetails"
    output_tokens: int
    total_tokens: int


class ConversationInputTokenDetails(BaseModel):
    """
    Represents the input token details of the conversation.
    """

    audio_tokens: int
    cached_tokens: int
    text_tokens: int
    cached_tokens_details: "ConversationInputCachedTokensDetails"


class ConversationInputCachedTokensDetails(BaseModel):
    """
    Represents the cached token details of the conversation.
    """

    text_tokens: int
    audio_tokens: int


class ConversationOutputTokenDetails(BaseModel):
    """
    Represents the output token details of the conversation.
    """

    audio_tokens: int
    text_tokens: int
