from rasa_sdk import Action
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk import Tracker
from typing import Any, Text, Dict, List
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from rasa_sdk.events import AllSlotsReset

class SentimentAnalysisAction(Action):
    def name(self) -> Text:
        return "action_analyze_sentiment"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> List[Dict[Text, Any]]:
        # Get the user's first and last descriptions from the slots
        first_description = tracker.get_slot("first_description")
        last_description = tracker.get_slot("last_description")
        third_description = tracker.get_slot("third_description")
        fourth_description = tracker.get_slot("fourth_description")
        fifth_description = tracker.get_slot("fifth_description")
        sixth_description = tracker.get_slot("sixth_description")

        # Perform sentiment analysis
        analyzer = SentimentIntensityAnalyzer()
        first_sentiment = analyzer.polarity_scores(first_description)
        last_sentiment = analyzer.polarity_scores(last_description)
        third_sentiment = analyzer.polarity_scores(third_description)
        fourth_sentiment = analyzer.polarity_scores(fourth_description)
        fifth_sentiment = analyzer.polarity_scores(fifth_description)
        sixth_sentiment = analyzer.polarity_scores(sixth_description)

        # Calculate the overall sentiment score
        overall_sentiment_score = (first_sentiment["compound"] + last_sentiment["compound"] + third_sentiment["compound"] + fourth_sentiment["compound"] + fifth_sentiment["compound"] + sixth_sentiment["compound"])

        # Determine the overall sentiment
        overall_sentiment = "positive" if overall_sentiment_score >= 0 else "negative"

        # Send the appropriate response based on the sentiment analysis result
        if overall_sentiment == "positive":
            message = "Your child's responses indicate a positive outlook. They seem to be in good spirits and engaging positively with the content."
        else:
            message = "It may be helpful to pay attention to your child's emotional state. Their responses suggest they might be feeling a bit down or troubled. Offering support and attention could be beneficial."

        # Send the sentiment analysis result to the user
        dispatcher.utter_message(text=message)

        return []



class ActionResetSlots(Action):
    def name(self) -> Text:
        return "action_reset_slots"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return [AllSlotsReset()]