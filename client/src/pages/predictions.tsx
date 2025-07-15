import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Brain, TrendingUp, AlertTriangle } from "lucide-react";
import { api } from "@/lib/api";
import { Link } from "wouter";

export default function Predictions() {
  const { data: riskPredictions, isLoading: riskLoading } = useQuery({
    queryKey: ["/api/predictions/risk"],
    queryFn: () => api.getRiskPredictions(),
  });

  const { data: bedOptimizations, isLoading: bedLoading } = useQuery({
    queryKey: ["/api/predictions/bed-optimization"],
    queryFn: () => api.getBedOptimizations(),
  });

  const { data: insights, isLoading: insightsLoading } = useQuery({
    queryKey: ["/api/llm/clinical-insights"],
    queryFn: () => api.getClinicalInsights(),
  });

  const getRiskLevel = (score: number) => {
    if (score >= 70) return { level: "high", color: "destructive", text: "High Risk" };
    if (score >= 40) return { level: "medium", color: "warning", text: "Medium Risk" };
    return { level: "low", color: "secondary", text: "Low Risk" };
  };

  if (riskLoading && bedLoading && insightsLoading) {
    return (
      <div className="space-y-6">
        <h1 className="text-3xl font-bold">AI Predictions</h1>
        <div className="text-center py-8">Loading predictions...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">AI Predictions</h1>
        <div className="flex items-center space-x-2">
          <Brain className="h-5 w-5 text-accent" />
          <span className="text-sm text-muted-foreground">
            Powered by Advanced Machine Learning
          </span>
        </div>
      </div>

      {/* Risk Predictions Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5 text-destructive" />
              <span>Patient Risk Predictions</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {!riskPredictions || riskPredictions.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No risk predictions available
              </div>
            ) : (
              <div className="space-y-4">
                {riskPredictions.map((prediction) => {
                  const risk = getRiskLevel(prediction.riskScore);
                  return (
                    <div
                      key={prediction.patientId}
                      className={`p-4 rounded-lg border transition-all duration-200 hover:shadow-sm risk-card ${risk.level}`}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-3">
                          <div className={`w-3 h-3 rounded-full bg-${risk.color} ${risk.level === 'high' ? 'animate-pulse' : ''}`} />
                          <div>
                            <h3 className="font-semibold text-foreground">
                              Patient {prediction.patientId}
                            </h3>
                            <p className="text-sm text-muted-foreground">
                              Confidence: {Math.round(prediction.confidence * 100)}%
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <Badge variant={risk.color as any} className="mb-1">
                            {prediction.riskScore}%
                          </Badge>
                          <p className="text-xs text-muted-foreground">{risk.text}</p>
                        </div>
                      </div>
                      
                      <Progress value={prediction.riskScore} className="mb-3" />
                      
                      <div className="space-y-2">
                        <p className="text-sm font-medium text-foreground">Risk Factors:</p>
                        <ul className="text-sm text-muted-foreground space-y-1">
                          {prediction.factors.map((factor, index) => (
                            <li key={index} className="flex items-center space-x-2">
                              <div className="w-1 h-1 bg-muted-foreground rounded-full" />
                              <span>{factor}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      
                      <div className="mt-3 p-2 bg-background rounded border">
                        <p className="text-sm font-medium text-foreground mb-1">Recommendation:</p>
                        <p className="text-sm text-muted-foreground">{prediction.recommendation}</p>
                      </div>
                      
                      <div className="mt-3 text-right">
                        <Link href={`/patients/${prediction.patientId}`}>
                          <span className="text-sm text-primary hover:underline cursor-pointer">
                            View Patient Details →
                          </span>
                        </Link>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Summary Stats */}
        <Card>
          <CardHeader>
            <CardTitle>Prediction Summary</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {riskPredictions && (
              <>
                <div className="text-center">
                  <div className="text-2xl font-bold text-foreground">{riskPredictions.length}</div>
                  <p className="text-sm text-muted-foreground">Total Predictions</p>
                </div>
                
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">High Risk</span>
                    <Badge variant="destructive">
                      {riskPredictions.filter(p => p.riskScore >= 70).length}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Medium Risk</span>
                    <Badge className="bg-warning text-warning-foreground">
                      {riskPredictions.filter(p => p.riskScore >= 40 && p.riskScore < 70).length}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Low Risk</span>
                    <Badge variant="secondary">
                      {riskPredictions.filter(p => p.riskScore < 40).length}
                    </Badge>
                  </div>
                </div>
                
                <div className="pt-2 border-t">
                  <div className="text-center">
                    <div className="text-lg font-bold text-accent">
                      {Math.round(riskPredictions.reduce((sum, p) => sum + p.confidence, 0) / riskPredictions.length * 100)}%
                    </div>
                    <p className="text-sm text-muted-foreground">Average Confidence</p>
                  </div>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Bed Optimization Recommendations */}
      {bedOptimizations && bedOptimizations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5 text-secondary" />
              <span>Bed Allocation Optimization</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {bedOptimizations.map((opt, index) => (
                <div key={index} className="p-4 bg-green-50 rounded-lg border border-green-200">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold text-foreground">Patient {opt.patientId}</h3>
                    <Badge variant="outline">Priority {opt.priority}</Badge>
                  </div>
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">
                      <span className="font-medium">Current:</span> {opt.currentBed}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      <span className="font-medium">Recommended:</span> {opt.recommendedBed}
                    </p>
                    <p className="text-sm text-foreground">{opt.reason}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* AI Clinical Insights */}
      {insights && insights.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Brain className="h-5 w-5 text-accent" />
              <span>AI Clinical Insights</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {insights.map((insight, index) => {
                const colorClass = insight.type === "staffing" 
                  ? "from-blue-50 to-purple-50 border-blue-200"
                  : "from-green-50 to-blue-50 border-green-200";
                
                return (
                  <div
                    key={index}
                    className={`p-4 bg-gradient-to-r rounded-lg border ${colorClass}`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <h3 className="font-semibold text-foreground">{insight.title}</h3>
                      <Badge 
                        variant={insight.priority === "high" ? "destructive" : 
                                insight.priority === "medium" ? "secondary" : "outline"}
                      >
                        {insight.priority} priority
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">{insight.content}</p>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
