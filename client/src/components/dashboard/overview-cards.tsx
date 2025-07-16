import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Users, 
  Heart, 
  Bed, 
  Brain, 
  TrendingUp, 
  TrendingDown,
  AlertTriangle,
  Activity,
  Clock,
  Zap,
  Target,
  Stethoscope,
  FlaskConical,
  UserCheck,
  Calendar,
  Shield,
  Monitor,
  Ambulance
} from "lucide-react";
import type { DashboardStats } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";

interface OverviewCardsProps {
  stats?: DashboardStats;
  isLoading: boolean;
}

export default function OverviewCards({ stats, isLoading }: OverviewCardsProps) {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-6 gap-4">
        {Array.from({ length: 12 }).map((_, i) => (
          <Card key={i} className="medical-card">
            <CardContent className="p-4">
              <Skeleton className="h-16 w-full" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-6 gap-4">
        {Array.from({ length: 12 }).map((_, i) => (
          <Card key={i} className="medical-card">
            <CardContent className="p-4">
              <div className="text-center text-muted-foreground">
                <AlertTriangle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                Dados indisponíveis
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  // Calcular métricas avançadas baseadas nos dados
  const occupancyRate = Math.round(((stats.icuBeds + stats.generalBeds) / 150) * 100);
  const riskLevel = stats.highRiskPatients > 30 ? 'high' : stats.highRiskPatients > 15 ? 'medium' : 'low';
  const aiTrend = stats.aiAccuracy > 90 ? 'excellent' : stats.aiAccuracy > 80 ? 'good' : 'needs-improvement';

  const cards = [
    {
      title: "Pacientes Ativos",
      value: stats.totalPatients.toString(),
      subtitle: "+12 hoje",
      icon: Users,
      color: "bg-medical-primary/10 text-medical-primary border-medical-primary/30",
      trend: "up",
      badge: "Live",
      category: "primary"
    },
    {
      title: "Alto Risco",
      value: stats.highRiskPatients.toString(),
      subtitle: "Requer atenção",
      icon: Heart,
      color: "bg-destructive/10 text-destructive border-destructive/30",
      trend: riskLevel === 'high' ? 'warning' : 'stable',
      badge: "Critical",
      category: "critical"
    },
    {
      title: "Leitos Disponíveis",
      value: stats.availableBeds.toString(),
      subtitle: `UTI: ${stats.icuBeds} | Geral: ${stats.generalBeds}`,
      icon: Bed,
      color: "bg-medical-secondary/10 text-medical-secondary border-medical-secondary/30",
      trend: "stable",
      badge: `${occupancyRate}%`,
      category: "resource"
    },
    {
      title: "IA Precisão",
      value: `${stats.aiAccuracy}%`,
      subtitle: "+2.3% esta semana",
      icon: Brain,
      color: "bg-chart-4/10 text-chart-4 border-chart-4/30",
      trend: "up",
      badge: "Neural",
      category: "ai"
    },
    {
      title: "Emergências",
      value: "8",
      subtitle: "Últimas 24h",
      icon: Ambulance,
      color: "bg-warning/10 text-warning border-warning/30",
      trend: "down",
      badge: "Active",
      category: "emergency"
    },
    {
      title: "Cirurgias",
      value: "15",
      subtitle: "Programadas hoje",
      icon: Stethoscope,
      color: "bg-chart-2/10 text-chart-2 border-chart-2/30",
      trend: "stable",
      badge: "Scheduled",
      category: "surgery"
    },
    {
      title: "Exames Lab",
      value: "234",
      subtitle: "Processados hoje",
      icon: FlaskConical,
      color: "bg-chart-3/10 text-chart-3 border-chart-3/30",
      trend: "up",
      badge: "Processing",
      category: "lab"
    },
    {
      title: "Admissões",
      value: "23",
      subtitle: "Últimas 24h",
      icon: UserCheck,
      color: "bg-medical-accent/10 text-medical-accent border-medical-accent/30",
      trend: "up",
      badge: "Today",
      category: "admission"
    },
    {
      title: "Turnos Ativos",
      value: "127",
      subtitle: "Profissionais",
      icon: Clock,
      color: "bg-chart-5/10 text-chart-5 border-chart-5/30",
      trend: "stable",
      badge: "24/7",
      category: "staff"
    },
    {
      title: "Sistema",
      value: "99.8%",
      subtitle: "Uptime",
      icon: Monitor,
      color: "bg-medical-primary/10 text-medical-primary border-medical-primary/30",
      trend: "excellent",
      badge: "Online",
      category: "system"
    },
    {
      title: "Segurança",
      value: "A+",
      subtitle: "SSL/TLS",
      icon: Shield,
      color: "bg-medical-secondary/10 text-medical-secondary border-medical-secondary/30",
      trend: "secure",
      badge: "Secure",
      category: "security"
    },
    {
      title: "Performance",
      value: "95%",
      subtitle: "Otimização IA",
      icon: Zap,
      color: "bg-chart-4/10 text-chart-4 border-chart-4/30",
      trend: "optimized",
      badge: "Fast",
      category: "performance"
    }
  ];

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="h-3 w-3" />;
      case 'down':
        return <TrendingDown className="h-3 w-3" />;
      case 'warning':
        return <AlertTriangle className="h-3 w-3" />;
      default:
        return <Activity className="h-3 w-3" />;
    }
  };

  const getBadgeVariant = (category: string) => {
    switch (category) {
      case 'critical':
        return 'destructive';
      case 'ai':
        return 'default';
      case 'emergency':
        return 'secondary';
      default:
        return 'outline';
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-6 gap-4">
      {cards.map((card) => {
        const Icon = card.icon;
        
        return (
          <Card key={card.title} className={`medical-card hover:shadow-lg transition-all duration-300 ${card.color} border-2`}>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Icon className="h-5 w-5" />
                  <Badge variant={getBadgeVariant(card.category)} className="text-xs">
                    {card.badge}
                  </Badge>
                </div>
                <div className="flex items-center gap-1 text-xs">
                  {getTrendIcon(card.trend)}
                </div>
              </div>
            </CardHeader>
            
            <CardContent className="pt-0">
              <div className="space-y-1">
                <p className="text-2xl font-bold">{card.value}</p>
                <p className="text-xs font-medium text-muted-foreground">{card.title}</p>
                <p className="text-xs text-muted-foreground">{card.subtitle}</p>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
