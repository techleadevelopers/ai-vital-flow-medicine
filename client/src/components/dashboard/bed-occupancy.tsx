import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";

export default function BedOccupancy() {
  const { data: beds, isLoading } = useQuery({
    queryKey: ["/api/beds"],
    queryFn: () => api.getBeds(),
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Bed Occupancy Status</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-16 w-full" />
          ))}
        </CardContent>
      </Card>
    );
  }

  if (!beds) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Bed Occupancy Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground py-8">
            Unable to load bed occupancy data
          </div>
        </CardContent>
      </Card>
    );
  }

  // Calculate occupancy by type
  const bedTypes = ["ICU", "General", "Emergency"];
  const occupancyData = bedTypes.map(type => {
    const typeBeds = beds.filter(bed => bed.type === type);
    const occupiedBeds = typeBeds.filter(bed => bed.status === "occupied");
    const total = typeBeds.length;
    const occupied = occupiedBeds.length;
    const percentage = total > 0 ? Math.round((occupied / total) * 100) : 0;
    
    return {
      type,
      occupied,
      total,
      percentage,
      available: total - occupied
    };
  });

  const getProgressColor = (percentage: number) => {
    if (percentage >= 80) return "bg-destructive";
    if (percentage >= 60) return "bg-warning";
    return "bg-secondary";
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Bed Occupancy Status</CardTitle>
          <Button variant="outline" size="sm" className="text-accent">
            <RefreshCw className="h-4 w-4 mr-2" />
            Optimize Allocation
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {occupancyData.map((data) => (
          <div key={data.type}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-foreground">{data.type}</span>
              <span className="text-sm text-muted-foreground">
                {data.occupied}/{data.total} beds occupied
              </span>
            </div>
            <Progress 
              value={data.percentage} 
              className="h-3 mb-1"
            />
            <p className="text-xs text-muted-foreground">
              {data.percentage}% occupancy
            </p>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
