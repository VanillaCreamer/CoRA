# This code is clone from peft third package and being changed for personalization
__version__ = "0.3.0.dev0"
from minigpt4.models.ppeft.pmapping import get_peft_model
from minigpt4.models.ppeft.plora import PLoraConfig, PLoraModel
from minigpt4.models.ppeft.pconfig import PeftType
from minigpt4.models.ppeft.ppeft_model import PeftModel
