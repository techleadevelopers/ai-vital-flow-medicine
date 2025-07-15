import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import type { PatientFlowData } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";

interface PatientFlowChartProps {
  data?: PatientFlowData;
  isLoading: boolean;
}

export default function PatientFlowChart({ data, isLoading }: PatientFlowChartProps) {
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Patient Flow Analytics</CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-64 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Patient Flow Analytics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-center justify-center text-muted-foreground">
            Unable to load patient flow data
          </div>
        </CardContent>
      </Card>
    );
  }

  // Transform data for the chart
  const chartData = data.admissions.map((admission, index) => ({
    time: `${index.toString().padStart(2, '0')}:00`,
    admissions: admission,
    discharges: data.discharges[index] || 0
  }));

  return (
    <Card className="cyberpunk-card">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-primary neon-text">Patient Flow Analytics</CardTitle>
            <p className="text-accent text-sm">Last 24 hours patient admission and discharge trends</p>
          </div>
          <div className="flex space-x-2">
            <Button variant="default" size="sm" className="cyberpunk-button">24H</Button>
            <Button variant="outline" size="sm" className="border-primary/50 text-accent hover:bg-primary/10">7D</Button>
            <Button variant="outline" size="sm" className="border-primary/50 text-accent hover:bg-primary/10">30D</Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis 
                dataKey="time" 
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
              />
              <YAxis 
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: "hsl(var(--background))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "var(--radius)"
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="admissions" 
                stroke="hsl(var(--primary))" 
                strokeWidth={2}
                name="Admissions"
                dot={{ fill: "hsl(var(--primary))", strokeWidth: 2 }}
              />
              <Line 
                type="monotone" 
                dataKey="discharges" 
                stroke="hsl(var(--secondary))" 
                strokeWidth={2}
                name="Discharges"
                dot={{ fill: "hsl(var(--secondary))", strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
