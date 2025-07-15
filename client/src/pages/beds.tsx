import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Bed, RefreshCw, Filter } from "lucide-react";
import { api } from "@/lib/api";
import { useState } from "react";

export default function Beds() {
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [typeFilter, setTypeFilter] = useState<string>("all");

  const { data: beds, isLoading } = useQuery({
    queryKey: ["/api/beds"],
    queryFn: () => api.getBeds(),
  });

  const { data: optimizations, isLoading: optimizationsLoading } = useQuery({
    queryKey: ["/api/predictions/bed-optimization"],
    queryFn: () => api.getBedOptimizations(),
  });

  const filteredBeds = beds?.filter(bed => {
    const statusMatch = statusFilter === "all" || bed.status === statusFilter;
    const typeMatch = typeFilter === "all" || bed.type === typeFilter;
    return statusMatch && typeMatch;
  }) || [];

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "occupied":
        return <Badge variant="destructive">Occupied</Badge>;
      case "available":
        return <Badge variant="secondary">Available</Badge>;
      case "maintenance":
        return <Badge className="bg-warning text-warning-foreground">Maintenance</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const getTypeBadge = (type: string) => {
    switch (type) {
      case "ICU":
        return <Badge variant="destructive">ICU</Badge>;
      case "General":
        return <Badge variant="default">General</Badge>;
      case "Emergency":
        return <Badge className="bg-warning text-warning-foreground">Emergency</Badge>;
      default:
        return <Badge variant="outline">{type}</Badge>;
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <h1 className="text-3xl font-bold">Bed Management</h1>
        <div className="text-center py-8">Loading beds...</div>
      </div>
    );
  }

  const bedStats = beds ? {
    total: beds.length,
    occupied: beds.filter(b => b.status === "occupied").length,
    available: beds.filter(b => b.status === "available").length,
    maintenance: beds.filter(b => b.status === "maintenance").length,
  } : { total: 0, occupied: 0, available: 0, maintenance: 0 };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Bed Management</h1>
        <Button className="bg-accent hover:bg-accent/90">
          <RefreshCw className="h-4 w-4 mr-2" />
          Optimize Allocation
        </Button>
      </div>

      {/* Bed Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6 text-center">
            <div className="text-2xl font-bold text-foreground">{bedStats.total}</div>
            <p className="text-sm text-muted-foreground">Total Beds</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <div className="text-2xl font-bold text-destructive">{bedStats.occupied}</div>
            <p className="text-sm text-muted-foreground">Occupied</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <div className="text-2xl font-bold text-secondary">{bedStats.available}</div>
            <p className="text-sm text-muted-foreground">Available</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <div className="text-2xl font-bold text-warning">{bedStats.maintenance}</div>
            <p className="text-sm text-muted-foreground">Maintenance</p>
          </CardContent>
        </Card>
      </div>

      {/* Optimization Recommendations */}
      {optimizations && optimizations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>AI Optimization Recommendations</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {optimizations.slice(0, 3).map((opt, index) => (
                <div key={index} className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-foreground">
                        Patient {opt.patientId}: {opt.currentBed} → {opt.recommendedBed}
                      </p>
                      <p className="text-sm text-muted-foreground">{opt.reason}</p>
                    </div>
                    <Button size="sm" variant="outline">Apply</Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Filters and Bed Grid */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Bed Status Overview</CardTitle>
            <div className="flex items-center space-x-2">
              <Filter className="h-4 w-4 text-muted-foreground" />
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger className="w-32">
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="occupied">Occupied</SelectItem>
                  <SelectItem value="available">Available</SelectItem>
                  <SelectItem value="maintenance">Maintenance</SelectItem>
                </SelectContent>
              </Select>
              <Select value={typeFilter} onValueChange={setTypeFilter}>
                <SelectTrigger className="w-32">
                  <SelectValue placeholder="Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="ICU">ICU</SelectItem>
                  <SelectItem value="General">General</SelectItem>
                  <SelectItem value="Emergency">Emergency</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {filteredBeds.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No beds found matching the selected filters.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {filteredBeds.map((bed) => (
                <Card key={bed.id} className="hover:shadow-md transition-shadow">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <Bed className="h-5 w-5 text-muted-foreground" />
                        <span className="font-semibold">{bed.bedNumber}</span>
                      </div>
                      {getStatusBadge(bed.status)}
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Room:</span>
                        <span className="text-sm font-medium">{bed.roomNumber}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Type:</span>
                        {getTypeBadge(bed.type)}
                      </div>
                      {bed.patientId && (
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Patient:</span>
                          <span className="text-sm font-medium">{bed.patientId}</span>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
