import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

export default function VitalSignsChart() {
  // Mock vital signs data for demonstration
  const vitalSignsData = [
    { time: "10:00", heartRate: 72, bloodPressure: 120, temperature: 36.5, oxygenSat: 98 },
    { time: "11:00", heartRate: 74, bloodPressure: 122, temperature: 36.6, oxygenSat: 97 },
    { time: "12:00", heartRate: 76, bloodPressure: 125, temperature: 36.8, oxygenSat: 96 },
    { time: "13:00", heartRate: 73, bloodPressure: 121, temperature: 36.7, oxygenSat: 98 },
    { time: "14:00", heartRate: 75, bloodPressure: 123, temperature: 36.6, oxygenSat: 97 },
    { time: "15:00", heartRate: 74, bloodPressure: 120, temperature: 36.5, oxygenSat: 98 },
  ];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Real-time Vital Signs</CardTitle>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-secondary rounded-full animate-pulse" />
            <Badge variant="outline" className="text-xs">Live</Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={vitalSignsData}>
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
                dataKey="heartRate" 
                stroke="hsl(var(--destructive))" 
                strokeWidth={2}
                name="Heart Rate (BPM)"
                dot={{ fill: "hsl(var(--destructive))", strokeWidth: 2 }}
              />
              <Line 
                type="monotone" 
                dataKey="bloodPressure" 
                stroke="hsl(var(--accent))" 
                strokeWidth={2}
                name="Blood Pressure (Systolic)"
                dot={{ fill: "hsl(var(--accent))", strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
