import { Bell, User } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";

export default function Header() {
  return (
    <header className="cyberpunk-card border-b border-primary/30 px-6 py-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-primary neon-text">Healthcare Dashboard</h2>
          <p className="text-accent">Real-time patient monitoring and optimization</p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="relative">
            <Bell className="h-5 w-5 text-accent hover:text-primary transition-colors cursor-pointer" />
            <Badge className="absolute -top-1 -right-1 h-3 w-3 p-0 flex items-center justify-center bg-destructive text-black animate-pulse">
              <span className="text-xs">3</span>
            </Badge>
          </div>
          <div className="flex items-center space-x-2">
            <Avatar className="h-8 w-8 border border-primary/50">
              <AvatarFallback className="bg-primary/20 text-primary border border-primary/30">
                <User className="h-4 w-4" />
              </AvatarFallback>
            </Avatar>
            <span className="text-foreground font-medium">Dr. Sarah Johnson</span>
          </div>
        </div>
      </div>
    </header>
  );
}
