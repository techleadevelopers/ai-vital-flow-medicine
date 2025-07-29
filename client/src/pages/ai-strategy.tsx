import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Brain, 
  Zap, 
  Target, 
  TrendingUp, 
  Users, 
  DollarSign, 
  Shield, 
  CheckCircle,
  Clock,
  ArrowRight,
  Rocket,
  Award,
  Building
} from 'lucide-react';

interface StrategyPhase {
  id: string;
  title: string;
  description: string;
  duration: string;
  investment: string;
  roi: string;
  status: 'planned' | 'in-progress' | 'completed';
  milestones: string[];
  technologies: string[];
}

interface AICapability {
  name: string;
  description: string;
  maturity: number;
  impact: 'high' | 'medium' | 'low';
  timeline: string;
  investment: string;
}

const strategyPhases: StrategyPhase[] = [
  {
    id: 'phase1',
    title: 'Fase 1: Fundação IA',
    description: 'Implementação das capacidades básicas de IA e infraestrutura',
    duration: '6 meses',
    investment: 'R$ 2.4M',
    roi: '185%',
    status: 'completed',
    milestones: [
      'Implementação do sistema de predição de risco',
      'Otimização de leitos com ML',
      'Dashboard analytics em tempo real',
      'Treinamento básico da equipe'
    ],
    technologies: ['Neural Networks', 'Random Forest', 'Time Series', 'Real-time Analytics']
  },
  {
    id: 'phase2',
    title: 'Fase 2: IA Avançada',
    description: 'Causal AI, Digital Twins e sistemas de decisão autônoma',
    duration: '8 meses',
    investment: 'R$ 3.8M',
    roi: '240%',
    status: 'in-progress',
    milestones: [
      'Sistema de IA Causal implementado',
      'Gêmeos digitais do hospital',
      'Integração IoT completa',
      'Automação de processos críticos'
    ],
    technologies: ['Causal AI', 'Digital Twins', 'IoT', 'Reinforcement Learning']
  },
  {
    id: 'phase3',
    title: 'Fase 3: Autonomia Total',
    description: 'Sistema completamente autônomo com AGI médica',
    duration: '12 meses',
    investment: 'R$ 5.2M',
    roi: '320%',
    status: 'planned',
    milestones: [
      'AGI médica implementada',
      'Robôs cirúrgicos autônomos',
      'Diagnóstico automático',
      'Gestão hospitalar autônoma'
    ],
    technologies: ['AGI', 'Robotic Surgery', 'Computer Vision', 'Autonomous Systems']
  }
];

const aiCapabilities: AICapability[] = [
  {
    name: 'Predição de Deterioração',
    description: 'Sistema neural para prever deterioração clínica',
    maturity: 94,
    impact: 'high',
    timeline: 'Implementado',
    investment: 'R$ 450k'
  },
  {
    name: 'Otimização de Recursos',
    description: 'RL para alocação dinâmica de recursos',
    maturity: 87,
    impact: 'high',
    timeline: 'Q1 2025',
    investment: 'R$ 380k'
  },
  {
    name: 'Causal AI',
    description: 'Identificação de fatores causais e intervenções',
    maturity: 72,
    impact: 'high',
    timeline: 'Q2 2025',
    investment: 'R$ 520k'
  },
  {
    name: 'Digital Twins',
    description: 'Simulação digital completa do hospital',
    maturity: 68,
    impact: 'high',
    timeline: 'Q2 2025',
    investment: 'R$ 640k'
  },
  {
    name: 'IoT Integration',
    description: 'Integração completa com dispositivos médicos',
    maturity: 85,
    impact: 'medium',
    timeline: 'Q1 2025',
    investment: 'R$ 320k'
  },
  {
    name: 'AGI Médica',
    description: 'Inteligência artificial geral para medicina',
    maturity: 15,
    impact: 'high',
    timeline: 'Q4 2025',
    investment: 'R$ 1.2M'
  }
];

export default function AIStrategy() {
  const [selectedPhase, setSelectedPhase] = useState<StrategyPhase | null>(null);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800';
      case 'in-progress': return 'bg-blue-100 text-blue-800';
      case 'planned': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="h-4 w-4" />;
      case 'in-progress': return <Clock className="h-4 w-4" />;
      case 'planned': return <Target className="h-4 w-4" />;
      default: return <Clock className="h-4 w-4" />;
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'bg-red-100 text-red-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Estratégia de IA</h1>
          <p className="text-muted-foreground">
            Roadmap estratégico para implementação de IA hospitalar avançada
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline">
            <TrendingUp className="mr-2 h-4 w-4" />
            Análise ROI
          </Button>
          <Button>
            <Rocket className="mr-2 h-4 w-4" />
            Plano Detalhado
          </Button>
        </div>
      </div>

      <Tabs defaultValue="roadmap" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="roadmap">Roadmap</TabsTrigger>
          <TabsTrigger value="capabilities">Capacidades</TabsTrigger>
          <TabsTrigger value="roi">ROI & Investimento</TabsTrigger>
          <TabsTrigger value="implementation">Implementação</TabsTrigger>
        </TabsList>

        <TabsContent value="roadmap" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {strategyPhases.map((phase) => (
              <Card 
                key={phase.id}
                className={`cursor-pointer transition-all hover:shadow-lg ${
                  selectedPhase?.id === phase.id ? 'ring-2 ring-blue-500' : ''
                }`}
                onClick={() => setSelectedPhase(phase)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <Badge className={getStatusColor(phase.status)}>
                      {getStatusIcon(phase.status)}
                      <span className="ml-1">{phase.status}</span>
                    </Badge>
                    <div className="text-sm font-medium">{phase.duration}</div>
                  </div>
                  <CardTitle className="text-lg">{phase.title}</CardTitle>
                  <CardDescription>{phase.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Investimento</span>
                      <span className="font-medium">{phase.investment}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">ROI Esperado</span>
                      <span className="font-medium text-green-600">{phase.roi}</span>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {phase.milestones.length} marcos principais
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {selectedPhase && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Building className="h-5 w-5" />
                  <span>Detalhes da Fase: {selectedPhase.title}</span>
                </CardTitle>
                <CardDescription>
                  {selectedPhase.description}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <h4 className="font-semibold mb-2">Duração</h4>
                      <div className="text-2xl font-bold">{selectedPhase.duration}</div>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-2">Investimento</h4>
                      <div className="text-2xl font-bold text-blue-600">{selectedPhase.investment}</div>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-2">ROI</h4>
                      <div className="text-2xl font-bold text-green-600">{selectedPhase.roi}</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-semibold mb-3">Marcos Principais</h4>
                      <ul className="space-y-2">
                        {selectedPhase.milestones.map((milestone, index) => (
                          <li key={index} className="flex items-start space-x-2">
                            <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                            <span className="text-sm">{milestone}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-3">Tecnologias</h4>
                      <div className="flex flex-wrap gap-2">
                        {selectedPhase.technologies.map((tech, index) => (
                          <Badge key={index} variant="secondary">
                            {tech}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="capabilities" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {aiCapabilities.map((capability) => (
              <Card key={capability.name}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{capability.name}</CardTitle>
                    <Badge className={getImpactColor(capability.impact)}>
                      {capability.impact}
                    </Badge>
                  </div>
                  <CardDescription>{capability.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>Maturidade</span>
                        <span className="font-medium">{capability.maturity}%</span>
                      </div>
                      <Progress value={capability.maturity} className="h-2" />
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <div className="text-muted-foreground">Timeline</div>
                        <div className="font-medium">{capability.timeline}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Investimento</div>
                        <div className="font-medium">{capability.investment}</div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="roi" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center space-x-2">
                  <DollarSign className="h-5 w-5" />
                  <span>Investimento Total</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">R$ 11.4M</div>
                <p className="text-sm text-muted-foreground">Próximos 26 meses</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center space-x-2">
                  <TrendingUp className="h-5 w-5" />
                  <span>ROI Acumulado</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-600">248%</div>
                <p className="text-sm text-muted-foreground">Estimativa conservadora</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center space-x-2">
                  <Target className="h-5 w-5" />
                  <span>Economia Anual</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">R$ 8.7M</div>
                <p className="text-sm text-muted-foreground">Projeção ano 3</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center space-x-2">
                  <Clock className="h-5 w-5" />
                  <span>Payback</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">18 meses</div>
                <p className="text-sm text-muted-foreground">Break-even estimado</p>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Benefícios Quantificados</CardTitle>
              <CardDescription>
                Impacto financeiro detalhado da implementação de IA
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold mb-3">Economia Direta</h4>
                    <ul className="space-y-2 text-sm">
                      <li className="flex justify-between">
                        <span>Redução de reinternações</span>
                        <span className="font-medium">R$ 2.4M/ano</span>
                      </li>
                      <li className="flex justify-between">
                        <span>Otimização de pessoal</span>
                        <span className="font-medium">R$ 1.8M/ano</span>
                      </li>
                      <li className="flex justify-between">
                        <span>Eficiência operacional</span>
                        <span className="font-medium">R$ 1.6M/ano</span>
                      </li>
                      <li className="flex justify-between">
                        <span>Redução de desperdícios</span>
                        <span className="font-medium">R$ 980k/ano</span>
                      </li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-3">Benefícios Indiretos</h4>
                    <ul className="space-y-2 text-sm">
                      <li className="flex justify-between">
                        <span>Melhoria na satisfação</span>
                        <span className="font-medium">R$ 1.2M/ano</span>
                      </li>
                      <li className="flex justify-between">
                        <span>Redução de riscos</span>
                        <span className="font-medium">R$ 890k/ano</span>
                      </li>
                      <li className="flex justify-between">
                        <span>Compliance e auditoria</span>
                        <span className="font-medium">R$ 560k/ano</span>
                      </li>
                      <li className="flex justify-between">
                        <span>Vantagem competitiva</span>
                        <span className="font-medium">R$ 1.5M/ano</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="implementation" className="space-y-4">
          <Alert>
            <Rocket className="h-4 w-4" />
            <AlertDescription>
              <strong>Status Atual:</strong> Fase 1 completada com sucesso (ROI: 185%). 
              Fase 2 iniciada com 68% de progresso. Cronograma no prazo.
            </AlertDescription>
          </Alert>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Próximos Marcos</CardTitle>
                <CardDescription>
                  Principais entregas dos próximos 6 meses
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
                    <div>
                      <div className="font-medium">Sistema Causal AI</div>
                      <div className="text-sm text-muted-foreground">Março 2025</div>
                    </div>
                    <Badge>In Progress</Badge>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div>
                      <div className="font-medium">Digital Twins Hospital</div>
                      <div className="text-sm text-muted-foreground">Abril 2025</div>
                    </div>
                    <Badge variant="outline">Planned</Badge>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div>
                      <div className="font-medium">IoT Integration</div>
                      <div className="text-sm text-muted-foreground">Maio 2025</div>
                    </div>
                    <Badge variant="outline">Planned</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Recursos Necessários</CardTitle>
                <CardDescription>
                  Equipe e infraestrutura para implementação
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold mb-2">Equipe Técnica</h4>
                    <ul className="space-y-1 text-sm">
                      <li>• 3 Data Scientists Senior</li>
                      <li>• 2 ML Engineers</li>
                      <li>• 1 AI Research Lead</li>
                      <li>• 2 DevOps Engineers</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Infraestrutura</h4>
                    <ul className="space-y-1 text-sm">
                      <li>• GPU Cluster (8x RTX 4090)</li>
                      <li>• 256GB RAM por nó</li>
                      <li>• 10TB SSD Storage</li>
                      <li>• Rede 10Gbps</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}