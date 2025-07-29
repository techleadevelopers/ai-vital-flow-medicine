import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { 
  ArrowRight, 
  Brain, 
  GitBranch, 
  Lightbulb, 
  Network, 
  Target, 
  TrendingUp,
  Zap,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Info
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, Cell } from 'recharts';

interface CausalFactor {
  id: string;
  name: string;
  impact: number;
  confidence: number;
  category: 'clinical' | 'operational' | 'environmental' | 'behavioral';
  direction: 'positive' | 'negative' | 'neutral';
  evidence: string[];
}

interface CausalChain {
  id: string;
  title: string;
  factors: CausalFactor[];
  outcome: string;
  strength: number;
  validated: boolean;
}

interface Intervention {
  id: string;
  name: string;
  target: string;
  expectedImpact: number;
  cost: number;
  feasibility: number;
  timeframe: string;
  status: 'proposed' | 'testing' | 'implemented' | 'proven';
}

const causalFactors: CausalFactor[] = [
  {
    id: 'staffing-ratio',
    name: 'Proporção Enfermeiro/Paciente',
    impact: 85,
    confidence: 94,
    category: 'operational',
    direction: 'negative',
    evidence: ['Estudo Johns Hopkins 2019', 'Metanálise RN4CAST', 'Dados internos 2023-2024']
  },
  {
    id: 'early-warning',
    name: 'Sistema de Alerta Precoce',
    impact: 72,
    confidence: 89,
    category: 'clinical',
    direction: 'positive',
    evidence: ['RCT Medicine 2020', 'Implementação piloto Q1 2024']
  },
  {
    id: 'shift-handover',
    name: 'Qualidade da Passagem de Plantão',
    impact: 68,
    confidence: 91,
    category: 'operational',
    direction: 'positive',
    evidence: ['Análise de incidentes 2024', 'Survey satisfação equipe']
  },
  {
    id: 'ambient-noise',
    name: 'Ruído Ambiental UTI',
    impact: 45,
    confidence: 78,
    category: 'environmental',
    direction: 'negative',
    evidence: ['Estudo sleep quality ICU', 'Monitoramento IoT']
  }
];

const causalChains: CausalChain[] = [
  {
    id: 'deterioration-chain',
    title: 'Deterioração Clínica → Mortalidade',
    factors: causalFactors.slice(0, 3),
    outcome: 'Redução 23% mortalidade',
    strength: 87,
    validated: true
  },
  {
    id: 'readmission-chain',
    title: 'Reinternação → Custos',
    factors: causalFactors.slice(1, 4),
    outcome: 'Economia R$ 2.4M/ano',
    strength: 76,
    validated: false
  }
];

const interventions: Intervention[] = [
  {
    id: 'increase-staffing',
    name: 'Aumento Equipe Noturna',
    target: 'Proporção Enfermeiro/Paciente',
    expectedImpact: 65,
    cost: 180000,
    feasibility: 78,
    timeframe: '3 meses',
    status: 'proposed'
  },
  {
    id: 'ai-early-warning',
    name: 'IA Alerta Precoce Avançado',
    target: 'Sistema de Alerta Precoce',
    expectedImpact: 82,
    cost: 45000,
    feasibility: 95,
    timeframe: '6 semanas',
    status: 'testing'
  },
  {
    id: 'structured-handover',
    name: 'Protocolo SBAR Digitalizado',
    target: 'Qualidade da Passagem de Plantão',
    expectedImpact: 58,
    cost: 12000,
    feasibility: 92,
    timeframe: '4 semanas',
    status: 'implemented'
  }
];

const impactPredictionData = [
  { week: 1, baseline: 100, intervention: 98, predicted: 95 },
  { week: 2, baseline: 100, intervention: 94, predicted: 89 },
  { week: 4, baseline: 100, intervention: 87, predicted: 82 },
  { week: 8, baseline: 100, intervention: 79, predicted: 75 },
  { week: 12, baseline: 100, intervention: 72, predicted: 68 },
  { week: 16, baseline: 100, intervention: 68, predicted: 63 },
  { week: 24, baseline: 100, intervention: 65, predicted: 60 }
];

export default function CausalAI() {
  const [selectedChain, setSelectedChain] = useState<CausalChain | null>(null);
  const [selectedIntervention, setSelectedIntervention] = useState<Intervention | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const runCausalAnalysis = () => {
    setIsAnalyzing(true);
    setTimeout(() => setIsAnalyzing(false), 4000);
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'clinical': return 'bg-red-100 text-red-800';
      case 'operational': return 'bg-blue-100 text-blue-800';
      case 'environmental': return 'bg-green-100 text-green-800';
      case 'behavioral': return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getDirectionIcon = (direction: string) => {
    switch (direction) {
      case 'positive': return <TrendingUp className="h-4 w-4 text-green-500" />;
      case 'negative': return <TrendingUp className="h-4 w-4 text-red-500 rotate-180" />;
      default: return <ArrowRight className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'proposed': return <Info className="h-4 w-4 text-blue-500" />;
      case 'testing': return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'implemented': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'proven': return <CheckCircle className="h-4 w-4 text-green-600" />;
      default: return <XCircle className="h-4 w-4 text-gray-500" />;
    }
  };

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">IA Causal</h1>
          <p className="text-muted-foreground">
            Descobrindo relações causais e otimizando intervenções baseadas em evidências
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button onClick={runCausalAnalysis} disabled={isAnalyzing}>
            {isAnalyzing ? (
              <>
                <Brain className="mr-2 h-4 w-4 animate-pulse" />
                Analisando...
              </>
            ) : (
              <>
                <Network className="mr-2 h-4 w-4" />
                Análise Causal
              </>
            )}
          </Button>
        </div>
      </div>

      <Tabs defaultValue="factors" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="factors">Fatores Causais</TabsTrigger>
          <TabsTrigger value="chains">Cadeias Causais</TabsTrigger>
          <TabsTrigger value="interventions">Intervenções</TabsTrigger>
          <TabsTrigger value="validation">Validação</TabsTrigger>
        </TabsList>

        <TabsContent value="factors" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {causalFactors.map((factor) => (
              <Card key={factor.id} className="hover:shadow-lg transition-shadow">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      {getDirectionIcon(factor.direction)}
                      <CardTitle className="text-lg">{factor.name}</CardTitle>
                    </div>
                    <Badge className={getCategoryColor(factor.category)}>
                      {factor.category}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <div className="text-sm text-muted-foreground mb-1">Impacto</div>
                        <div className="flex items-center space-x-2">
                          <Progress value={factor.impact} className="flex-1 h-2" />
                          <span className="text-sm font-medium">{factor.impact}%</span>
                        </div>
                      </div>
                      <div>
                        <div className="text-sm text-muted-foreground mb-1">Confiança</div>
                        <div className="flex items-center space-x-2">
                          <Progress value={factor.confidence} className="flex-1 h-2" />
                          <span className="text-sm font-medium">{factor.confidence}%</span>
                        </div>
                      </div>
                    </div>
                    
                    <div>
                      <div className="text-sm text-muted-foreground mb-2">Evidências</div>
                      <ul className="space-y-1">
                        {factor.evidence.map((evidence, index) => (
                          <li key={index} className="text-xs text-muted-foreground flex items-center">
                            <CheckCircle className="h-3 w-3 mr-1 text-green-500" />
                            {evidence}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="chains" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {causalChains.map((chain) => (
              <Card 
                key={chain.id}
                className={`cursor-pointer transition-all hover:shadow-lg ${
                  selectedChain?.id === chain.id ? 'ring-2 ring-blue-500' : ''
                }`}
                onClick={() => setSelectedChain(chain)}
              >
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{chain.title}</CardTitle>
                    <div className="flex items-center space-x-2">
                      {chain.validated ? (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 text-yellow-500" />
                      )}
                      <Badge variant="secondary">{chain.strength}%</Badge>
                    </div>
                  </div>
                  <CardDescription>{chain.outcome}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="text-sm text-muted-foreground">
                      Fatores envolvidos: {chain.factors.length}
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {chain.factors.map((factor) => (
                        <Badge key={factor.id} variant="outline" className="text-xs">
                          {factor.name}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {selectedChain && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <GitBranch className="h-5 w-5" />
                  <span>Análise da Cadeia Causal</span>
                </CardTitle>
                <CardDescription>
                  {selectedChain.title} - Força: {selectedChain.strength}%
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div className="flex items-center space-x-4">
                    {selectedChain.factors.map((factor, index) => (
                      <div key={factor.id} className="flex items-center space-x-2">
                        <div className="flex flex-col items-center">
                          <div className="w-3 h-3 bg-blue-500 rounded-full" />
                          <div className="text-xs text-center mt-1 max-w-16">
                            {factor.name.split(' ')[0]}
                          </div>
                        </div>
                        {index < selectedChain.factors.length - 1 && (
                          <ArrowRight className="h-4 w-4 text-gray-400" />
                        )}
                      </div>
                    ))}
                    <div className="flex items-center space-x-2">
                      <ArrowRight className="h-4 w-4 text-gray-400" />
                      <div className="flex flex-col items-center">
                        <Target className="h-6 w-6 text-green-500" />
                        <div className="text-xs text-center mt-1">
                          Resultado
                        </div>
                      </div>
                    </div>
                  </div>

                  <Separator />

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-semibold mb-2">Fatores Principais</h4>
                      <div className="space-y-2">
                        {selectedChain.factors.map((factor) => (
                          <div key={factor.id} className="flex items-center justify-between text-sm">
                            <span>{factor.name}</span>
                            <span className="font-medium">{factor.impact}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-2">Métricas</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span>Força da Evidência</span>
                          <span className="font-medium">{selectedChain.strength}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Status</span>
                          <span className="font-medium">
                            {selectedChain.validated ? 'Validado' : 'Em Teste'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Impacto Esperado</span>
                          <span className="font-medium">{selectedChain.outcome}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="interventions" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {interventions.map((intervention) => (
              <Card 
                key={intervention.id}
                className={`cursor-pointer transition-all hover:shadow-lg ${
                  selectedIntervention?.id === intervention.id ? 'ring-2 ring-green-500' : ''
                }`}
                onClick={() => setSelectedIntervention(intervention)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{intervention.name}</CardTitle>
                    <div className="flex items-center space-x-1">
                      {getStatusIcon(intervention.status)}
                      <Badge variant="secondary">{intervention.status}</Badge>
                    </div>
                  </div>
                  <CardDescription>{intervention.target}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span>Impacto Esperado</span>
                      <span className="font-medium">{intervention.expectedImpact}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Custo</span>
                      <span className="font-medium">R$ {intervention.cost.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Viabilidade</span>
                      <span className="font-medium">{intervention.feasibility}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Prazo</span>
                      <span className="font-medium">{intervention.timeframe}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {selectedIntervention && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Lightbulb className="h-5 w-5" />
                  <span>Análise de Intervenção</span>
                </CardTitle>
                <CardDescription>
                  {selectedIntervention.name} - Impacto: {selectedIntervention.expectedImpact}%
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <h4 className="font-semibold mb-2">Impacto Esperado</h4>
                      <div className="space-y-2">
                        <Progress value={selectedIntervention.expectedImpact} className="h-2" />
                        <p className="text-sm text-muted-foreground">
                          {selectedIntervention.expectedImpact}% de melhoria no {selectedIntervention.target}
                        </p>
                      </div>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-2">Viabilidade</h4>
                      <div className="space-y-2">
                        <Progress value={selectedIntervention.feasibility} className="h-2" />
                        <p className="text-sm text-muted-foreground">
                          {selectedIntervention.feasibility}% de probabilidade de sucesso
                        </p>
                      </div>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-2">ROI Estimado</h4>
                      <div className="space-y-2">
                        <div className="text-2xl font-bold text-green-600">
                          R$ {(selectedIntervention.expectedImpact * 1000).toLocaleString()}
                        </div>
                        <p className="text-sm text-muted-foreground">
                          Economia anual estimada
                        </p>
                      </div>
                    </div>
                  </div>

                  <Separator />

                  <div>
                    <h4 className="font-semibold mb-4">Predição de Impacto ao Longo do Tempo</h4>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={impactPredictionData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="week" />
                        <YAxis />
                        <Tooltip />
                        <Line type="monotone" dataKey="baseline" stroke="#9ca3af" strokeWidth={2} strokeDasharray="5 5" />
                        <Line type="monotone" dataKey="intervention" stroke="#3b82f6" strokeWidth={2} />
                        <Line type="monotone" dataKey="predicted" stroke="#10b981" strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="validation" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Validação Experimental</CardTitle>
                <CardDescription>
                  Resultados de testes controlados e estudos piloto
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-5 w-5 text-green-600" />
                      <span className="font-medium">Alerta Precoce IA</span>
                    </div>
                    <Badge className="bg-green-100 text-green-800">Validado</Badge>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-yellow-50 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <AlertTriangle className="h-5 w-5 text-yellow-600" />
                      <span className="font-medium">Protocolo SBAR</span>
                    </div>
                    <Badge className="bg-yellow-100 text-yellow-800">Em Teste</Badge>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <Info className="h-5 w-5 text-blue-600" />
                      <span className="font-medium">Escala Noturna</span>
                    </div>
                    <Badge className="bg-blue-100 text-blue-800">Planejado</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Métricas de Validação</CardTitle>
                <CardDescription>
                  Indicadores de sucesso e eficácia das intervenções
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">-23%</div>
                      <div className="text-sm text-muted-foreground">Mortalidade</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">+34%</div>
                      <div className="text-sm text-muted-foreground">Detecção Precoce</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600">-18%</div>
                      <div className="text-sm text-muted-foreground">Reinternações</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-orange-600">-2.3h</div>
                      <div className="text-sm text-muted-foreground">Tempo Resposta</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Alert>
            <Zap className="h-4 w-4" />
            <AlertDescription>
              <strong>Próximos Passos:</strong> Implementar sistema de feedback automático 
              para validação contínua das intervenções causais em tempo real.
            </AlertDescription>
          </Alert>
        </TabsContent>
      </Tabs>
    </div>
  );
}