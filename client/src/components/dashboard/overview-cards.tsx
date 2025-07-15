import { Card, CardContent } from "@/components/ui/card";
import { Users, Heart, Bed, Brain, TrendingUp } from "lucide-react";
import type { DashboardStats } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";

interface OverviewCardsProps {
  stats?: DashboardStats;
  isLoading: boolean;
}

export default function OverviewCards({ stats, isLoading }: OverviewCardsProps) {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i} className="p-6">
            <Skeleton className="h-20 w-full" />
          </Card>
        ))}
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i} className="p-6">
            <CardContent className="pt-6">
              <div className="text-center text-muted-foreground">
                Unable to load dashboard statistics
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  const cards = [
    {
      title: "Total Patients",
      value: stats.totalPatients.toString(),
      subtitle: "+12 today",
      icon: Users,
      color: "bg-primary/10 text-primary border-primary/30",
      trend: "up"
    },
    {
      title: "High Risk Patients",
      value: stats.highRiskPatients.toString(),
      subtitle: "Requires attention",
      icon: Heart,
      color: "bg-destructive/10 text-destructive border-destructive/30",
      trend: "warning"
    },
    {
      title: "Available Beds",
      value: stats.availableBeds.toString(),
      subtitle: `ICU: ${stats.icuBeds} | General: ${stats.generalBeds}`,
      icon: Bed,
      color: "bg-accent/10 text-accent border-accent/30",
      trend: "stable"
    },
    {
      title: "AI Accuracy",
      value: `${stats.aiAccuracy}%`,
      subtitle: "+2.3% this week",
      icon: Brain,
      color: "bg-secondary/10 text-secondary border-secondary/30",
      trend: "up"
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {cards.map((card, index) => {
        const Icon = card.icon;
        return (
          <Card key={index} className="cyberpunk-card hover:cyber-glow transition-all duration-300">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-accent text-sm font-medium">{card.title}</p>
                  <p className="text-3xl font-bold text-primary neon-text">{card.value}</p>
                  <p className="text-sm mt-1 flex items-center">
                    {card.trend === "up" && <TrendingUp className="h-3 w-3 mr-1 text-secondary" />}
                    <span className={card.trend === "warning" ? "text-destructive" : "text-accent"}>
                      {card.subtitle}
                    </span>
                  </p>
                </div>
                <div className={`w-12 h-12 rounded-lg flex items-center justify-center border ${card.color}`}>
                  <Icon className="h-6 w-6" />
                </div>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
