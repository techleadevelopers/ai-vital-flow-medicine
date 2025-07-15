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
