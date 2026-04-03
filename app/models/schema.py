# from pydantic import BaseModel
# from typing import Optional, List, Dict

# class KPIResponse(BaseModel):
#     total_revenue: float
#     total_opex: float
#     total_energy_cost: float
#     total_profit: float
#     avg_productivity: float
#     avg_energy_efficiency: float
#     avg_cost_efficiency: float
#     avg_diesel_dependency: float
#     avg_utilization: float

# class PlotResponse(BaseModel):
#     cost_vs_revenue: str
#     energy_vs_revenue: str
#     utilization_vs_revenue: str
#     diesel_vs_grid: str
#     opex_breakdown: str

# class AnalysisResponse(BaseModel):
#     kpis: KPIResponse
#     plots: PlotResponse
#     ai_recommendations: Optional[str] = None
#     rev_score: Optional[float] = None
#     cost_score: Optional[float] = None
#     clf_acc: Optional[float] = None

# class PredictResponse(BaseModel):
#     predicted_revenue: Optional[float] = None
#     predicted_opex: Optional[float] = None
#     productivity_label: Optional[str] = None

# class ChatResponse(BaseModel):
#     reply: str