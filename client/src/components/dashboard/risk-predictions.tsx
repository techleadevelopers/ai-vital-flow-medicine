import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Brain } from "lucide-react";
import { api } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { Link } from "wouter";

export default function RiskPredictions() {
  const { data: predictions, isLoading } = useQuery({
    queryKey: ["/api/predictions/risk"],
    queryFn: () => api.getRiskPredictions(),
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            AI Risk Predictions
            <Brain className="h-5 w-5 text-accent" />
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-16 w-full" />
          ))}
        </CardContent>
      </Card>
    );
  }

  if (!predictions || predictions.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            AI Risk Predictions
            <Brain className="h-5 w-5 text-accent" />
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground py-8">
            No risk predictions available
          </div>
        </CardContent>
      </Card>
    );
  }

  const topPredictions = predictions
    .sort((a, b) => b.riskScore - a.riskScore)
    .slice(0, 3);

  const getRiskLevel = (score: number) => {
    if (score >= 70) return { level: "high", color: "destructive" };
    if (score >= 40) return { level: "medium", color: "warning" };
    return { level: "low", color: "secondary" };
  };

  return (
    <Card className="cyberpunk-card">
      <CardHeader>
        <CardTitle className="flex items-center justify-between text-primary neon-text">
          AI Risk Predictions
          <Brain className="h-5 w-5 text-accent" />
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {topPredictions.map((prediction) => {
          const risk = getRiskLevel(prediction.riskScore);
          return (
            <div
              key={prediction.patientId}
              className={`flex items-center justify-between p-3 rounded-lg border transition-all duration-200 hover:shadow-sm risk-card ${risk.level}`}
            >
              <div className="flex items-center space-x-3">
                <div className={`w-2 h-2 rounded-full bg-${risk.color} ${risk.level === 'high' ? 'animate-pulse' : ''}`} />
                <div>
                  <p className="font-medium text-foreground">Patient {prediction.patientId}</p>
                  <p className="text-xs text-muted-foreground">
                    Confidence: {Math.round(prediction.confidence * 100)}%
                  </p>
                </div>
              </div>
              <div className="text-right">
                <Badge variant={risk.color as any} className="mb-1">
                  {prediction.riskScore}%
                </Badge>
                <p className="text-xs text-muted-foreground">Risk</p>
              </div>
            </div>
          );
        })}
        <Link href="/predictions">
          <Button className="w-full mt-4">
            View All Predictions
          </Button>
        </Link>
      </CardContent>
    </Card>
  );
}
