import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { UserPlus, AlertTriangle, CheckCircle, Activity } from "lucide-react";
import { api } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";

export default function RecentActivities() {
  const { data: activities, isLoading } = useQuery({
    queryKey: ["/api/activities"],
    queryFn: () => api.getActivities(5),
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Recent Activities</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-16 w-full" />
          ))}
        </CardContent>
      </Card>
    );
  }

  if (!activities || activities.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Recent Activities</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground py-8">
            No recent activities
          </div>
        </CardContent>
      </Card>
    );
  }

  const getActivityIcon = (type: string) => {
    switch (type) {
      case "admission":
        return UserPlus;
      case "alert":
        return AlertTriangle;
      case "discharge":
        return CheckCircle;
      default:
        return Activity;
    }
  };

  const getActivityColor = (severity: string) => {
    switch (severity) {
      case "warning":
        return "bg-red-100 text-destructive";
      case "error":
        return "bg-red-100 text-destructive";
      default:
        return "bg-blue-100 text-primary";
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    
    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins} minutes ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)} hours ago`;
    return date.toLocaleDateString();
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Activities</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {activities.map((activity) => {
          const Icon = getActivityIcon(activity.type);
          const colorClass = getActivityColor(activity.severity);
          
          return (
            <div key={activity.id} className="flex items-start space-x-3">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${colorClass}`}>
                <Icon className="h-4 w-4" />
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium text-foreground">{activity.description}</p>
                <p className="text-xs text-muted-foreground">
                  {formatTimestamp(activity.timestamp)}
                </p>
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
