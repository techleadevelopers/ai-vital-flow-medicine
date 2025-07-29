import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import Dashboard from "@/pages/dashboard";
import Patients from "@/pages/patients";
import PatientDetails from "@/pages/patient-details";
import Beds from "@/pages/beds";
import Predictions from "@/pages/predictions";
import Analytics from "@/pages/analytics";
import RiskAnalysis from "@/pages/risk-analysis";
import FlowPrediction from "@/pages/flow-prediction";
import Reports from "@/pages/reports";
import Admissions from "@/pages/admissions";
import Emergency from "@/pages/emergency";
import Discharges from "@/pages/discharges";
import Anomalies from "@/pages/anomalies";
import DigitalTwin from "@/pages/digital-twin";
import CausalAI from "@/pages/causal-ai";
import IoTIntegration from "@/pages/iot-integration";
import ExecutiveDashboard from "@/pages/executive-dashboard";
import AIStrategy from "@/pages/ai-strategy";
import NotFound from "@/pages/not-found";
import Sidebar from "@/components/layout/sidebar";
import Header from "@/components/layout/header";

function Router() {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 overflow-hidden">
        <Header />
        <div className="p-6 overflow-y-auto h-[calc(100vh-4rem)]">
          <Switch>
            <Route path="/" component={Dashboard} />
            <Route path="/dashboard" component={Dashboard} />
            <Route path="/patients" component={Patients} />
            <Route path="/patients/:id" component={PatientDetails} />
            <Route path="/beds" component={Beds} />
            <Route path="/predictions" component={Predictions} />
            <Route path="/analytics" component={Analytics} />
            <Route path="/risk-analysis" component={RiskAnalysis} />
            <Route path="/flow-prediction" component={FlowPrediction} />
            <Route path="/reports" component={Reports} />
            <Route path="/admissions" component={Admissions} />
            <Route path="/emergency" component={Emergency} />
            <Route path="/discharges" component={Discharges} />
            <Route path="/anomalies" component={Anomalies} />
            <Route path="/digital-twin" component={DigitalTwin} />
            <Route path="/causal-ai" component={CausalAI} />
            <Route path="/iot-integration" component={IoTIntegration} />
            <Route path="/executive-dashboard" component={ExecutiveDashboard} />
            <Route path="/ai-strategy" component={AIStrategy} />
            <Route component={NotFound} />
          </Switch>
        </div>
      </main>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Router />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
