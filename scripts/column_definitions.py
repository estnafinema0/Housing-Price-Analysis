from typing import List

def get_continuous_columns() -> List[str]:
    """Get the list of continuous columns used in the housing price prediction."""
    return [
        'Lot_Frontage', 'Lot_Area', 'Year_Built', 'Year_Remod_Add',
        'Mas_Vnr_Area', 'BsmtFin_SF_1', 'BsmtFin_SF_2', 'Bsmt_Unf_SF',
        'Total_Bsmt_SF', 'First_Flr_SF', 'Second_Flr_SF', 'Low_Qual_Fin_SF',
        'Gr_Liv_Area', 'Bsmt_Full_Bath', 'Bsmt_Half_Bath', 'Full_Bath',
        'Half_Bath', 'Bedroom_AbvGr', 'Kitchen_AbvGr', 'TotRms_AbvGrd',
        'Fireplaces', 'Garage_Cars', 'Garage_Area', 'Wood_Deck_SF',
        'Open_Porch_SF', 'Enclosed_Porch', 'Three_season_porch',
        'Screen_Porch', 'Pool_Area', 'Misc_Val', 'Mo_Sold', 'Year_Sold',
        'Longitude', 'Latitude'
    ]

def get_categorical_columns() -> List[str]:
    """Get the list of categorical columns used in the housing price prediction."""
    return [
        "Overall_Qual",
        "Garage_Qual",
        "Sale_Condition",
        "MS_Zoning"
    ] 