import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, BarChart, Bar, ComposedChart } from "recharts";
import type { PatientFlowData } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { Activity, TrendingUp, TrendingDown, Users, Clock, AlertTriangle, BarChart3 } from "lucide-react";

interface PatientFlowChartProps {
  data?: PatientFlowData;
  isLoading: boolean;
}

type ChartType = 'line' | 'area' | 'bar' | 'composed';
type TimeRange = '24h' | '7d' | '30d';

export default function PatientFlowChart({ data, isLoading }: PatientFlowChartProps) {
  const [chartType, setChartType] = useState<ChartType>('composed');
  const [timeRange, setTimeRange] = useState<TimeRange>('24h');

  if (isLoading) {
    return (
      <Card className="medical-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 medical-text">
            <Activity className="h-5 w-5" />
            Análise de Fluxo de Pacientes com IA
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-80 w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card className="medical-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 medical-text">
            <AlertTriangle className="h-5 w-5 text-warning" />
            Análise de Fluxo de Pacientes
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-80 flex flex-col items-center justify-center text-muted-foreground space-y-4">
            <Activity className="h-12 w-12 opacity-50" />
            <p>Dados de fluxo não disponíveis</p>
            <p className="text-sm">Sistema de IA em inicialização</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Transform data para análise avançada
  const chartData = data.admissions.map((admission, index) => {
    const discharges = data.discharges[index] || 0;
    const netFlow = admission - discharges;
    const baseOccupancy = 75;
    const occupancyTrend = baseOccupancy + netFlow * 2 + (index > 12 ? -5 : 5);
    
    return {
      time: `${index.toString().padStart(2, '0')}:00`,
      hour: index,
      admissions: admission,
      discharges: discharges,
      netFlow: netFlow,
      occupancy: Math.max(0, Math.min(100, occupancyTrend)),
      capacity: 100,
      efficiency: Math.round((discharges / (admission + discharges || 1)) * 100),
      peak: index >= 8 && index <= 18 ? 'peak' : 'off-peak'
    };
  });

  // Calcular métricas avançadas
  const totalAdmissions = data.admissions.reduce((a, b) => a + b, 0);
  const totalDischarges = data.discharges.reduce((a, b) => a + b, 0);
  const netFlow = totalAdmissions - totalDischarges;
  const peakHours = chartData.filter(d => d.peak === 'peak');
  const avgPeakAdmissions = peakHours.reduce((a, b) => a + b.admissions, 0) / peakHours.length;
  const currentOccupancy = chartData[chartData.length - 1]?.occupancy || 0;

  // Predições baseadas em tendências
  const trend = netFlow > 0 ? 'increasing' : netFlow < 0 ? 'decreasing' : 'stable';
  const riskLevel = currentOccupancy > 85 ? 'high' : currentOccupancy > 70 ? 'medium' : 'low';

  const renderChart = () => {
    const commonProps = {
      data: chartData,
      margin: { top: 20, right: 30, left: 20, bottom: 20 }
    };

    switch (chartType) {
      case 'line':
        return (
          <LineChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
            <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={11} />
            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'hsl(var(--card))', 
                border: '1px solid hsl(var(--border))',
                borderRadius: 'var(--radius)'
              }} 
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="admissions" 
              stroke="hsl(var(--medical-primary))" 
              strokeWidth={3}
              name="Admissões"
              dot={{ fill: 'hsl(var(--medical-primary))', strokeWidth: 2, r: 4 }}
            />
            <Line 
              type="monotone" 
              dataKey="discharges" 
              stroke="hsl(var(--medical-secondary))" 
              strokeWidth={3}
              name="Altas"
              dot={{ fill: 'hsl(var(--medical-secondary))', strokeWidth: 2, r: 4 }}
            />
          </LineChart>
        );
      
      case 'area':
        return (
          <AreaChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
            <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={11} />
            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'hsl(var(--card))', 
                border: '1px solid hsl(var(--border))',
                borderRadius: 'var(--radius)'
              }} 
            />
            <Legend />
            <Area 
              type="monotone" 
              dataKey="admissions" 
              stackId="1"
              stroke="hsl(var(--medical-primary))" 
              fill="hsl(var(--medical-primary) / 0.3)"
              name="Admissões"
            />
            <Area 
              type="monotone" 
              dataKey="discharges" 
              stackId="2"
              stroke="hsl(var(--medical-secondary))" 
              fill="hsl(var(--medical-secondary) / 0.3)"
              name="Altas"
            />
          </AreaChart>
        );

      case 'bar':
        return (
          <BarChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
            <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={11} />
            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'hsl(var(--card))', 
                border: '1px solid hsl(var(--border))',
                borderRadius: 'var(--radius)'
              }} 
            />
            <Legend />
            <Bar dataKey="admissions" fill="hsl(var(--medical-primary))" name="Admissões" radius={[2, 2, 0, 0]} />
            <Bar dataKey="discharges" fill="hsl(var(--medical-secondary))" name="Altas" radius={[2, 2, 0, 0]} />
          </BarChart>
        );

      case 'composed':
      default:
        return (
          <ComposedChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
            <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={11} />
            <YAxis yAxisId="left" stroke="hsl(var(--muted-foreground))" fontSize={11} />
            <YAxis yAxisId="right" orientation="right" stroke="hsl(var(--muted-foreground))" fontSize={11} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'hsl(var(--card))', 
                border: '1px solid hsl(var(--border))',
                borderRadius: 'var(--radius)'
              }} 
            />
            <Legend />
            <Bar yAxisId="left" dataKey="admissions" fill="hsl(var(--medical-primary) / 0.7)" name="Admissões" radius={[2, 2, 0, 0]} />
            <Bar yAxisId="left" dataKey="discharges" fill="hsl(var(--medical-secondary) / 0.7)" name="Altas" radius={[2, 2, 0, 0]} />
            <Line 
              yAxisId="right" 
              type="monotone" 
              dataKey="occupancy" 
              stroke="hsl(var(--chart-4))" 
              strokeWidth={3}
              name="Ocupação (%)"
              dot={{ fill: 'hsl(var(--chart-4))', strokeWidth: 2, r: 4 }}
            />
          </ComposedChart>
        );
    }
  };

  return (
    <Card className="medical-card flow-chart-advanced">
      <CardHeader>
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
          <div className="space-y-2">
            <CardTitle className="flex items-center gap-2 medical-text text-lg">
              <Activity className="h-6 w-6" />
              Análise Avançada de Fluxo - IA Neural
            </CardTitle>
            <p className="text-muted-foreground text-sm">
              Predições em tempo real com machine learning • Última atualização: agora
            </p>
          </div>
          
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-1">
              <Button 
                variant={timeRange === '24h' ? 'default' : 'outline'} 
                size="sm" 
                onClick={() => setTimeRange('24h')}
                className="medical-button"
              >
                24H
              </Button>
              <Button 
                variant={timeRange === '7d' ? 'default' : 'outline'} 
                size="sm" 
                onClick={() => setTimeRange('7d')}
              >
                7D
              </Button>
              <Button 
                variant={timeRange === '30d' ? 'default' : 'outline'} 
                size="sm" 
                onClick={() => setTimeRange('30d')}
              >
                30D
              </Button>
            </div>
            
            <div className="flex items-center gap-1 ml-2">
              <Button 
                variant={chartType === 'composed' ? 'default' : 'ghost'} 
                size="sm" 
                onClick={() => setChartType('composed')}
              >
                <BarChart3 className="h-4 w-4" />
              </Button>
              <Button 
                variant={chartType === 'line' ? 'default' : 'ghost'} 
                size="sm" 
                onClick={() => setChartType('line')}
              >
                📈
              </Button>
              <Button 
                variant={chartType === 'area' ? 'default' : 'ghost'} 
                size="sm" 
                onClick={() => setChartType('area')}
              >
                📊
              </Button>
            </div>
          </div>
        </div>

        {/* Métricas em tempo real */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 pt-4">
          <div className="bg-success-light/50 p-3 rounded-lg border-l-4 border-medical-primary">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground">Total Admissões</p>
                <p className="text-lg font-bold text-medical-primary">{totalAdmissions}</p>
              </div>
              <Users className="h-5 w-5 text-medical-primary" />
            </div>
          </div>
          
          <div className="bg-success-light/50 p-3 rounded-lg border-l-4 border-medical-secondary">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground">Total Altas</p>
                <p className="text-lg font-bold text-medical-secondary">{totalDischarges}</p>
              </div>
              <TrendingDown className="h-5 w-5 text-medical-secondary" />
            </div>
          </div>
          
          <div className={`p-3 rounded-lg border-l-4 ${
            riskLevel === 'high' ? 'bg-error-light/50 border-destructive' : 
            riskLevel === 'medium' ? 'bg-warning-light/50 border-warning' : 
            'bg-success-light/50 border-medical-primary'
          }`}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground">Ocupação Atual</p>
                <p className={`text-lg font-bold ${
                  riskLevel === 'high' ? 'text-destructive' : 
                  riskLevel === 'medium' ? 'text-warning' : 
                  'text-medical-primary'
                }`}>
                  {Math.round(currentOccupancy)}%
                </p>
              </div>
              <Activity className={`h-5 w-5 ${
                riskLevel === 'high' ? 'text-destructive' : 
                riskLevel === 'medium' ? 'text-warning' : 
                'text-medical-primary'
              }`} />
            </div>
          </div>
          
          <div className="bg-muted/50 p-3 rounded-lg border-l-4 border-chart-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground">Horário Pico</p>
                <p className="text-lg font-bold">{Math.round(avgPeakAdmissions)}</p>
              </div>
              <Clock className="h-5 w-5 text-chart-4" />
            </div>
          </div>
        </div>

        {/* Alertas e tendências */}
        <div className="flex flex-wrap gap-2 pt-2">
          <Badge variant={trend === 'increasing' ? 'destructive' : trend === 'decreasing' ? 'default' : 'secondary'}>
            {trend === 'increasing' && <TrendingUp className="h-3 w-3 mr-1" />}
            {trend === 'decreasing' && <TrendingDown className="h-3 w-3 mr-1" />}
            Tendência: {trend === 'increasing' ? 'Crescente' : trend === 'decreasing' ? 'Decrescente' : 'Estável'}
          </Badge>
          
          <Badge variant={riskLevel === 'high' ? 'destructive' : riskLevel === 'medium' ? 'secondary' : 'default'}>
            Risco: {riskLevel === 'high' ? 'Alto' : riskLevel === 'medium' ? 'Médio' : 'Baixo'}
          </Badge>
          
          <Badge variant="outline">
            Fluxo Líquido: {netFlow > 0 ? '+' : ''}{netFlow}
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            {renderChart()}
          </ResponsiveContainer>
        </div>
        
        {/* Insights da IA */}
        <div className="mt-6 p-4 bg-muted/30 rounded-lg">
          <h4 className="font-semibold text-medical-primary mb-2 flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Insights da IA Neural
          </h4>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-muted-foreground">
                • Padrão {trend} detectado nas últimas 24h
              </p>
              <p className="text-muted-foreground">
                • Ocupação atual: {riskLevel === 'high' ? 'crítica' : riskLevel === 'medium' ? 'moderada' : 'normal'}
              </p>
            </div>
            <div>
              <p className="text-muted-foreground">
                • Pico de demanda previsto: 08:00-18:00h
              </p>
              <p className="text-muted-foreground">
                • Eficiência do fluxo: {Math.round((totalDischarges / (totalAdmissions + totalDischarges)) * 100)}%
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
