import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";
import { 
  BarChart3, 
  Users, 
  Bed, 
  Brain, 
  Activity,
  Heart
} from "lucide-react";

const navigation = [
  { name: "Dashboard", href: "/dashboard", icon: BarChart3 },
  { name: "Patients", href: "/patients", icon: Users },
  { name: "Bed Management", href: "/beds", icon: Bed },
  { name: "AI Predictions", href: "/predictions", icon: Brain },
];

export default function Sidebar() {
  const [location] = useLocation();

  return (
    <aside className="w-64 cyberpunk-card border-r border-primary/30 shadow-xl">
      <div className="p-6 border-b border-primary/30">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-r from-primary to-accent rounded-lg flex items-center justify-center cyber-glow">
            <Heart className="text-black h-6 w-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-primary neon-text">VitalFlow AI</h1>
            <p className="text-sm text-accent">Healthcare Optimization</p>
          </div>
        </div>
      </div>
      
      <nav className="p-4 space-y-2">
        {navigation.map((item) => {
          const isActive = location === item.href || (item.href !== "/" && location.startsWith(item.href));
          const Icon = item.icon;
          
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "flex items-center space-x-3 px-4 py-3 rounded-lg font-medium transition-all duration-300",
                isActive
                  ? "cyberpunk-button bg-primary/20 text-primary border-primary/50"
                  : "text-muted-foreground hover:bg-primary/10 hover:text-primary hover:border-primary/30 border border-transparent"
              )}
            >
              <Icon className="h-5 w-5" />
              <span>{item.name}</span>
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
