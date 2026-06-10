from dataclasses import dataclass


@dataclass
class SurveyDataForElement:
    # Todo: go over all models
    #       have them be tagged by name or uid not both
    #       Preferably tag it by uid
    #       have a service to do the resolution
    name: str

    uid: str
    s: float
