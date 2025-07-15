import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Brain, Lightbulb, TrendingUp } from "lucide-react";
import { api } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";

export default function AIInsights() {
  const { data: insights, isLoading } = useQuery({
    queryKey: ["/api/llm/clinical-insights"],
    queryFn: () => api.getClinicalInsights(),
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            AI Clinical Insights
            <Brain className="h-5 w-5 text-accent" />
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {Array.from({ length: 2 }).map((_, i) => (
            <Skeleton key={i} className="h-20 w-full" />
          ))}
        </CardContent>
      </Card>
    );
  }

  if (!insights || insights.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            AI Clinical Insights
            <Brain className="h-5 w-5 text-accent" />
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground py-8">
            No insights available at this time
          </div>
        </CardContent>
      </Card>
    );
  }

  const getInsightIcon = (type: string) => {
    switch (type) {
      case "staffing":
        return Lightbulb;
      case "trend":
        return TrendingUp;
      default:
        return Brain;
    }
  };

  const getInsightColor = (type: string) => {
    switch (type) {
      case "staffing":
        return "from-blue-50 to-purple-50 border-blue-200";
      case "trend":
        return "from-green-50 to-blue-50 border-green-200";
      default:
        return "from-purple-50 to-blue-50 border-purple-200";
    }
  };

  const getPriorityBadge = (priority: string) => {
    switch (priority) {
      case "high":
        return <Badge variant="destructive">High Priority</Badge>;
      case "medium":
        return <Badge variant="secondary">Medium Priority</Badge>;
      default:
        return <Badge variant="outline">Low Priority</Badge>;
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          AI Clinical Insights
          <Brain className="h-5 w-5 text-accent" />
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {insights.slice(0, 2).map((insight, index) => {
          const Icon = getInsightIcon(insight.type);
          const colorClass = getInsightColor(insight.type);
          
          return (
            <div
              key={index}
              className={`p-4 bg-gradient-to-r rounded-lg border ${colorClass}`}
            >
              <div className="flex items-start space-x-3">
                <Icon className="text-accent mt-1 h-5 w-5" />
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <p className="text-sm font-medium text-foreground">{insight.title}</p>
                    {getPriorityBadge(insight.priority)}
                  </div>
                  <p className="text-sm text-muted-foreground">{insight.content}</p>
                </div>
              </div>
            </div>
          );
        })}
        <Button className="w-full mt-4 bg-accent hover:bg-accent/90">
          Generate Full Report
        </Button>
      </CardContent>
    </Card>
  );
}
