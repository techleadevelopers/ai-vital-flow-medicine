import { useQuery } from "@tanstack/react-query";
import { useRoute } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, Heart, Activity, Brain } from "lucide-react";
import { api } from "@/lib/api";
import { Link } from "wouter";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

export default function PatientDetails() {
  const [match, params] = useRoute("/patients/:id");
  const patientId = params?.id;

  const { data: patient, isLoading: patientLoading } = useQuery({
    queryKey: ["/api/patients", patientId],
    queryFn: () => api.getPatient(patientId!),
    enabled: !!patientId,
  });

  const { data: vitalSigns, isLoading: vitalSignsLoading } = useQuery({
    queryKey: ["/api/patients", patientId, "vital-signs"],
    queryFn: () => api.getPatientVitalSigns(patientId!),
    enabled: !!patientId,
  });

  const { data: riskPrediction, isLoading: riskLoading } = useQuery({
    queryKey: ["/api/predictions/risk", patientId],
    queryFn: () => api.getPatientRiskPrediction(patientId!),
    enabled: !!patientId,
  });

  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: ["/api/llm/summarize-patient", patientId],
    queryFn: () => api.getPatientSummary(patientId!),
    enabled: !!patientId,
  });

  if (!match || !patientId) {
    return <div>Patient not found</div>;
  }

  if (patientLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center space-x-4">
          <Link href="/patients">
            <Button variant="outline" size="sm">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Patients
            </Button>
          </Link>
        </div>
        <div className="text-center py-8">Loading patient details...</div>
      </div>
    );
  }

  if (!patient) {
    return (
      <div className="space-y-6">
        <div className="flex items-center space-x-4">
          <Link href="/patients">
            <Button variant="outline" size="sm">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Patients
            </Button>
          </Link>
        </div>
        <div className="text-center py-8 text-muted-foreground">
          Patient not found
        </div>
      </div>
    );
  }

  const getRiskBadge = (riskScore?: number) => {
    if (!riskScore) return <Badge variant="outline">Unknown</Badge>;
    if (riskScore >= 70) return <Badge variant="destructive">High Risk</Badge>;
    if (riskScore >= 40) return <Badge className="bg-warning text-warning-foreground">Medium Risk</Badge>;
    return <Badge variant="secondary">Low Risk</Badge>;
  };

  const formatDate = (date: Date | string) => {
    return new Date(date).toLocaleDateString();
  };

  // Transform vital signs for chart
  const chartData = vitalSigns?.map((vs, index) => ({
    time: new Date(vs.timestamp).toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit' 
    }),
    heartRate: vs.heartRate,
    bloodPressure: vs.bloodPressureSystolic,
    temperature: vs.temperature ? vs.temperature * 10 : null, // Scale for visibility
    oxygenSat: vs.oxygenSaturation,
  })) || [];

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center space-x-4">
        <Link href="/patients">
          <Button variant="outline" size="sm">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Patients
          </Button>
        </Link>
        <h1 className="text-3xl font-bold">Patient Details</h1>
      </div>

      {/* Patient Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Patient Information
              <Heart className="h-5 w-5 text-destructive" />
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">Name</p>
                <p className="font-semibold">{patient.name}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Patient ID</p>
                <p className="font-semibold">{patient.patientId}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Age</p>
                <p className="font-semibold">{patient.age}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Gender</p>
                <p className="font-semibold">{patient.gender}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Room</p>
                <p className="font-semibold">{patient.roomNumber}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Admission Date</p>
                <p className="font-semibold">{formatDate(patient.admissionDate)}</p>
              </div>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Diagnosis</p>
              <p className="font-semibold">{patient.diagnosis}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Comorbidities</p>
              <p className="font-semibold">{patient.comorbidities || "None"}</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Risk Assessment
              <Activity className="h-5 w-5 text-accent" />
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-foreground mb-2">
                {patient.riskScore || 0}%
              </div>
              {getRiskBadge(patient.riskScore)}
            </div>
            
            {!riskLoading && riskPrediction && (
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">Risk Factors:</p>
                <ul className="text-sm space-y-1">
                  {riskPrediction.factors.map((factor, index) => (
                    <li key={index} className="flex items-center space-x-2">
                      <div className="w-1 h-1 bg-muted-foreground rounded-full" />
                      <span>{factor}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {!riskLoading && riskPrediction && (
              <div>
                <p className="text-sm text-muted-foreground mb-1">Recommendation:</p>
                <p className="text-sm font-medium">{riskPrediction.recommendation}</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Vital Signs Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Vital Signs Trends</CardTitle>
        </CardHeader>
        <CardContent>
          {vitalSignsLoading ? (
            <div className="h-64 flex items-center justify-center">
              Loading vital signs...
            </div>
          ) : chartData.length === 0 ? (
            <div className="h-64 flex items-center justify-center text-muted-foreground">
              No vital signs data available
            </div>
          ) : (
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
                  <Line 
                    type="monotone" 
                    dataKey="heartRate" 
                    stroke="hsl(var(--destructive))" 
                    strokeWidth={2}
                    name="Heart Rate"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="bloodPressure" 
                    stroke="hsl(var(--primary))" 
                    strokeWidth={2}
                    name="Blood Pressure"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="oxygenSat" 
                    stroke="hsl(var(--secondary))" 
                    strokeWidth={2}
                    name="Oxygen Saturation"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </CardContent>
      </Card>

      {/* AI Clinical Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            AI Clinical Summary
            <Brain className="h-5 w-5 text-accent" />
          </CardTitle>
        </CardHeader>
        <CardContent>
          {summaryLoading ? (
            <div className="text-center py-4">Generating clinical summary...</div>
          ) : summary ? (
            <div className="p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border border-blue-200">
              <p className="text-sm text-foreground">{summary.summary}</p>
            </div>
          ) : (
            <div className="text-center py-4 text-muted-foreground">
              Clinical summary unavailable
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
