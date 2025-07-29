import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import vitalFlowLogo from "@/assets/images/l-Photoroom.png"; 
import { 
  BarChart3, Users, Bed, Brain, Activity, Heart, FileText,
  Shield, TrendingUp, AlertTriangle, Calendar, Clock, Stethoscope,
  Pill, FlaskConical, UserCheck, Settings, Database, Zap, Monitor,
  PieChart, LineChart, Building, Ambulance, ClipboardList,
  ChevronDown, ChevronRight, Microscope, Workflow, Target,
  Globe, BookOpen, HelpCircle
} from "lucide-react";
import { useState } from "react";

const navigationSections = [
  {
    title: "Painel Central",
    items: [
      { name: "Dashboard", href: "/dashboard", icon: BarChart3, badge: "Live" },
      { name: "Analytics", href: "/analytics", icon: TrendingUp, badge: "AI" },
      { name: "Relatórios", href: "/reports", icon: FileText, badge: "PDF" },
    ]
  },
  {
    title: "Gestão de Pacientes",
    items: [
      { name: "Pacientes", href: "/patients", icon: Users, badge: "387" },
      { name: "Admissões", href: "/admissions", icon: UserCheck, badge: "12" },
      { name: "Emergências", href: "/emergency", icon: Ambulance, badge: "3" },
      { name: "Altas", href: "/discharges", icon: ClipboardList },
    ]
  },
  {
    title: "Inteligência Artificial",
    items: [
      { name: "Predições IA", href: "/predictions", icon: Brain, badge: "Neural" },
      { name: "Análise de Risco", href: "/risk-analysis", icon: Target, badge: "ML" },
      { name: "Fluxo Preditivo", href: "/flow-prediction", icon: Workflow, badge: "LSTM" },
      { name: "Anomalias", href: "/anomalies", icon: AlertTriangle, badge: "2" },
    ]
  },
  {
    title: "IA Avançada",
    items: [
      { name: "IA Causal", href: "/causal-ai", icon: Brain, badge: "Causal" },
      { name: "Gêmeos Digitais", href: "/digital-twin", icon: Globe, badge: "Twin" },
      { name: "Integração IoT", href: "/iot-integration", icon: Zap, badge: "IoT" },
      { name: "Dashboard Executivo", href: "/executive-dashboard", icon: PieChart, badge: "C-Suite" },
      { name: "Estratégia de IA", href: "/ai-strategy", icon: Target, badge: "ROI" },
    ]
  },
  {
    title: "Recursos Hospitalares",
    items: [
      { name: "Leitos", href: "/beds", icon: Bed, badge: "98%" },
      { name: "Salas Cirúrgicas", href: "/surgery-rooms", icon: Stethoscope, badge: "6/8" },
      { name: "UTI", href: "/icu", icon: Heart, badge: "Critical" },
      { name: "Equipamentos", href: "/equipment", icon: Monitor, badge: "Active" },
    ]
  },
  {
    title: "Clínico & Laboratório",
    items: [
      { name: "Exames", href: "/lab-tests", icon: FlaskConical, badge: "24" },
      { name: "Medicamentos", href: "/medications", icon: Pill, badge: "Stock" },
      { name: "Patologia", href: "/pathology", icon: Microscope, badge: "7" },
      { name: "Diagnósticos", href: "/diagnostics", icon: Activity, badge: "AI" },
    ]
  },
  {
    title: "Operações",
    items: [
      { name: "Agenda", href: "/schedule", icon: Calendar, badge: "Today" },
      { name: "Turnos", href: "/shifts", icon: Clock, badge: "24/7" },
      { name: "Departamentos", href: "/departments", icon: Building },
      { name: "Workflows", href: "/workflows", icon: Workflow, badge: "Auto" },
    ]
  },
  {
    title: "Sistema",
    items: [
      { name: "Configurações", href: "/settings", icon: Settings },
      { name: "Segurança", href: "/security", icon: Shield, badge: "SSL" },
      { name: "Base de Dados", href: "/database", icon: Database, badge: "PG" },
      { name: "Performance", href: "/performance", icon: Zap, badge: "99.9%" },
    ]
  }
];

const quickStats = [
  { label: "Pacientes Ativos", value: "387", color: "bg-medical-primary" },
  { label: "Alto Risco", value: "23", color: "bg-destructive" },
  { label: "Leitos Disponíveis", value: "45", color: "bg-medical-secondary" },
  { label: "IA Precisão", value: "94%", color: "bg-chart-4" },
];

const scrollbarStyles = `
  .custom-scrollbar::-webkit-scrollbar {
    width: 8px;
  }
  .custom-scrollbar::-webkit-scrollbar-track {
    background: transparent;
  }
  .custom-scrollbar::-webkit-scrollbar-thumb {
    background: linear-gradient(
      to bottom,
      rgba(75, 248, 179, 0.28), /* Verde claro com opacidade */
      rgba(17, 128, 50, 0.6)  /* Azul claro com opacidade */
    );
    border-radius: 10px;
  }
  .custom-scrollbar::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(
      to bottom,
      rgba(16, 109, 75, 0.14), /* Verde claro mais sólido no hover */
      rgba(14, 153, 88, 0.27)
    );
  }
`;

export default function Sidebar() {
  const [location] = useLocation();
  const [expandedSections, setExpandedSections] = useState<string[]>([
    "Painel Central", 
    "Gestão de Pacientes", 
    "Inteligência Artificial"
  ]);

  const toggleSection = (sectionTitle: string) => {
    setExpandedSections(prev => 
      prev.includes(sectionTitle) 
        ? prev.filter(title => title !== sectionTitle)
        : [...prev, sectionTitle]
    );
  };

  return (
    <>
      <style>{scrollbarStyles}</style>
      <aside className="w-72 medical-card border-r border-border shadow-lg h-screen overflow-y-auto custom-scrollbar">
        {/* Header */}
        <div className="p-2 border-b border-border">
          <div className="flex items-center space-x-3">
            <img src={vitalFlowLogo} alt="VitalFlow AI Logo" className="w-60 h-30 object-contain" />
          </div>
        </div>

        {/* Quick Stats */}
        <div className="p-4 border-b border-border">
          <h3 className="text-xs font-semibold text-muted-foreground mb-3 uppercase tracking-wider">
            Status em Tempo Real
          </h3>
          <div className="grid grid-cols-2 gap-2">
            {quickStats.map((stat) => (
              <div key={stat.label} className="bg-muted/50 p-2 rounded-lg">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${stat.color}`} />
                  <span className="text-xs font-medium">{stat.value}</span>
                </div>
                <p className="text-xs text-muted-foreground mt-1">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
        
        {/* Navigation */}
        <nav className="p-4 space-y-2">
          {navigationSections.map((section) => {
            const isExpanded = expandedSections.includes(section.title);
            
            return (
              <div key={section.title} className="space-y-1">
                <button onClick={() => toggleSection(section.title)} className="w-full flex items-center justify-between px-3 py-2 text-sm font-medium text-muted-foreground hover:text-medical-primary transition-colors">
                  <span className="uppercase tracking-wider">{section.title}</span>
                  {isExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                </button>
                {isExpanded && (
                  <div className="space-y-1 ml-2">
                    {section.items.map((item) => {
                      const isActive = location === item.href || (item.href !== "/" && location.startsWith(item.href));
                      const Icon = item.icon;
                      return (
                        <Link key={item.name} href={item.href} className={cn( "flex items-center justify-between px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200", isActive ? "bg-medical-primary/10 text-medical-primary border-l-4 border-medical-primary" : "text-muted-foreground hover:bg-muted/50 hover:text-medical-primary border-l-4 border-transparent" )}>
                          <div className="flex items-center space-x-3">
                            <Icon className="h-4 w-4" />
                            <span>{item.name}</span>
                          </div>
                          {item.badge && (<Badge variant={isActive ? "default" : "secondary"} className="text-xs h-5">{item.badge}</Badge>)}
                        </Link>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-border mt-auto">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-medical-primary rounded-full animate-pulse" />
              <span>Sistema Online</span>
            </div>
            <div className="flex items-center gap-2">
              <Globe className="h-3 w-3" />
              <span>v2.1.0</span>
            </div>
          </div>
          <div className="mt-3 space-y-1">
            <Link href="/help" className="flex items-center gap-2 text-xs text-muted-foreground hover:text-medical-primary transition-colors">
              <HelpCircle className="h-3 w-3" />
              Ajuda & Suporte
            </Link>
            <Link href="/docs" className="flex items-center gap-2 text-xs text-muted-foreground hover:text-medical-primary transition-colors">
              <BookOpen className="h-3 w-3" />
              Documentação
            </Link>
          </div>
        </div>
      </aside>
    </>
  );
}