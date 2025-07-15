import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Search, User, Calendar } from "lucide-react";
import { api } from "@/lib/api";
import { Link } from "wouter";
import { useState } from "react";

export default function Patients() {
  const [searchTerm, setSearchTerm] = useState("");
  
  const { data: patients, isLoading } = useQuery({
    queryKey: ["/api/patients"],
    queryFn: () => api.getPatients(),
  });

  const filteredPatients = patients?.filter(patient =>
    patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    patient.patientId.toLowerCase().includes(searchTerm.toLowerCase()) ||
    patient.diagnosis?.toLowerCase().includes(searchTerm.toLowerCase())
  ) || [];

  const getRiskBadge = (riskScore?: number) => {
    if (!riskScore) return <Badge variant="outline">Unknown</Badge>;
    if (riskScore >= 70) return <Badge variant="destructive">High Risk</Badge>;
    if (riskScore >= 40) return <Badge className="bg-warning text-warning-foreground">Medium Risk</Badge>;
    return <Badge variant="secondary">Low Risk</Badge>;
  };

  const formatDate = (date: Date) => {
    return new Date(date).toLocaleDateString();
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold">Patients</h1>
        </div>
        <div className="text-center py-8">Loading patients...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Patients</h1>
        <Button>Add New Patient</Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Patient Management</CardTitle>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search patients by name, ID, or diagnosis..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
        </CardHeader>
        <CardContent>
          {filteredPatients.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              {searchTerm ? "No patients found matching your search." : "No patients available."}
            </div>
          ) : (
            <div className="space-y-4">
              {filteredPatients.map((patient) => (
                <div
                  key={patient.id}
                  className="flex items-center justify-between p-4 border rounded-lg hover:shadow-sm transition-shadow"
                >
                  <div className="flex items-center space-x-4">
                    <div className="w-12 h-12 bg-muted rounded-full flex items-center justify-center">
                      <User className="h-6 w-6 text-muted-foreground" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-foreground">{patient.name}</h3>
                      <p className="text-sm text-muted-foreground">ID: {patient.patientId}</p>
                      <p className="text-sm text-muted-foreground">
                        Age: {patient.age} | {patient.gender}
                      </p>
                    </div>
                  </div>
                  
                  <div className="text-center">
                    <p className="text-sm font-medium text-foreground">{patient.diagnosis}</p>
                    <p className="text-xs text-muted-foreground">Room {patient.roomNumber}</p>
                  </div>
                  
                  <div className="text-center">
                    <div className="flex items-center text-sm text-muted-foreground mb-1">
                      <Calendar className="h-4 w-4 mr-1" />
                      {formatDate(patient.admissionDate)}
                    </div>
                    {getRiskBadge(patient.riskScore)}
                  </div>
                  
                  <Link href={`/patients/${patient.patientId}`}>
                    <Button variant="outline">View Details</Button>
                  </Link>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
