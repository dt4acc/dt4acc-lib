from dataclasses import dataclass


@dataclass
class SurveyDataForElement:
    # Todo: go over all models
    #       have them return or uid not
    #       both
    #       Preferably uid
    #       have a service to do the resolution

    name: str

    uid: str
    s: float
