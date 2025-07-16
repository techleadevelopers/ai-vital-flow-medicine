import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import OverviewCards from "@/components/dashboard/overview-cards";
import PatientFlowChart from "@/components/charts/patient-flow-chart";
import RiskPredictions from "@/components/dashboard/risk-predictions";
import BedOccupancy from "@/components/dashboard/bed-occupancy";
import VitalSignsChart from "@/components/charts/vital-signs-chart";
import RecentActivities from "@/components/dashboard/recent-activities";
import AIInsights from "@/components/dashboard/ai-insights";
import AIPredictionsAdvanced from "@/components/dashboard/ai-predictions-advanced";

export default function Dashboard() {
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ["/api/dashboard/stats"],
    queryFn: () => api.getDashboardStats(),
  });

  const { data: patientFlow, isLoading: flowLoading } = useQuery({
    queryKey: ["/api/predictions/patient-flow"],
    queryFn: () => api.getPatientFlow(),
  });

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Overview Cards Enterprise */}
      <OverviewCards stats={stats} isLoading={statsLoading} />

      {/* AI Predictions Advanced Section */}
      <AIPredictionsAdvanced />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Patient Flow Chart */}
        <div className="lg:col-span-2">
          <PatientFlowChart data={patientFlow} isLoading={flowLoading} />
        </div>

        {/* Risk Predictions */}
        <RiskPredictions />
      </div>

      {/* Bed Management & Vital Signs */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <BedOccupancy />
        <VitalSignsChart />
      </div>

      {/* Activities & AI Insights */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RecentActivities />
        <AIInsights />
      </div>
    </div>
  );
}
