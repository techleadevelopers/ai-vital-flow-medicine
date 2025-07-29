import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { 
  Activity, 
  Battery, 
  Bluetooth, 
  Heart, 
  Router, 
  Smartphone, 
  Thermometer, 
  Wifi, 
  Zap,
  Watch,
  Stethoscope,
  Monitor,
  Gauge,
  Signal,
  AlertTriangle,
  CheckCircle,
  Clock
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

interface IoTDevice {
  id: string;
  name: string;
  type: 'wearable' | 'sensor' | 'monitor' | 'infusion';
  patientId?: string;
  status: 'online' | 'offline' | 'warning' | 'critical';
  batteryLevel: number;
  signalStrength: number;
  lastReading: string;
  location: string;
  metrics: {
    [key: string]: {
      value: number;
      unit: string;
      trend: 'up' | 'down' | 'stable';
      alert: boolean;
    };
  };
}

interface WearableData {
  timestamp: string;
  heartRate: number;
  steps: number;
  bloodPressure: { systolic: number; diastolic: number };
  oxygenSaturation: number;
  temperature: number;
  stress: number;
}

interface EnvironmentalSensor {
  id: string;
  location: string;
  temperature: number;
  humidity: number;
  airQuality: number;
  noiseLevel: number;
  lighting: number;
  pressure: number;
}

const iotDevices: IoTDevice[] = [
  {
    id: 'smartwatch-001',
    name: 'Apple Watch S9',
    type: 'wearable',
    patientId: 'P1847',
    status: 'online',
    batteryLevel: 78,
    signalStrength: 95,
    lastReading: '30 seg atrás',
    location: 'Leito 15A',
    metrics: {
      heartRate: { value: 72, unit: 'bpm', trend: 'stable', alert: false },
      steps: { value: 1234, unit: 'passos', trend: 'up', alert: false },
      oxygenSat: { value: 97, unit: '%', trend: 'stable', alert: false }
    }
  },
  {
    id: 'sensor-temp-002',
    name: 'Sensor Temperatura',
    type: 'sensor',
    patientId: 'P1823',
    status: 'online',
    batteryLevel: 45,
    signalStrength: 87,
    lastReading: '1 min atrás',
    location: 'UTI - Leito 3',
    metrics: {
      temperature: { value: 37.8, unit: '°C', trend: 'up', alert: true },
      humidity: { value: 45, unit: '%', trend: 'stable', alert: false }
    }
  },
  {
    id: 'monitor-cardiac-003',
    name: 'Monitor Cardíaco',
    type: 'monitor',
    patientId: 'P1901',
    status: 'warning',
    batteryLevel: 92,
    signalStrength: 76,
    lastReading: '15 seg atrás',
    location: 'Cardiologia',
    metrics: {
      heartRate: { value: 105, unit: 'bpm', trend: 'up', alert: true },
      bloodPressure: { value: 145, unit: 'mmHg', trend: 'up', alert: true }
    }
  },
  {
    id: 'infusion-pump-004',
    name: 'Bomba de Infusão',
    type: 'infusion',
    patientId: 'P1847',
    status: 'critical',
    batteryLevel: 12,
    signalStrength: 92,
    lastReading: '2 min atrás',
    location: 'UTI - Leito 1',
    metrics: {
      flow: { value: 0, unit: 'ml/h', trend: 'down', alert: true },
      pressure: { value: 85, unit: 'mmHg', trend: 'stable', alert: false }
    }
  }
];

const wearableData: WearableData[] = [
  {
    timestamp: '00:00',
    heartRate: 68,
    steps: 0,
    bloodPressure: { systolic: 120, diastolic: 80 },
    oxygenSaturation: 98,
    temperature: 36.5,
    stress: 15
  },
  {
    timestamp: '04:00',
    heartRate: 65,
    steps: 45,
    bloodPressure: { systolic: 118, diastolic: 78 },
    oxygenSaturation: 97,
    temperature: 36.3,
    stress: 12
  },
  {
    timestamp: '08:00',
    heartRate: 72,
    steps: 234,
    bloodPressure: { systolic: 125, diastolic: 82 },
    oxygenSaturation: 98,
    temperature: 36.7,
    stress: 25
  },
  {
    timestamp: '12:00',
    heartRate: 78,
    steps: 567,
    bloodPressure: { systolic: 130, diastolic: 85 },
    oxygenSaturation: 96,
    temperature: 36.8,
    stress: 35
  },
  {
    timestamp: '16:00',
    heartRate: 85,
    steps: 890,
    bloodPressure: { systolic: 135, diastolic: 88 },
    oxygenSaturation: 95,
    temperature: 37.1,
    stress: 45
  },
  {
    timestamp: '20:00',
    heartRate: 76,
    steps: 1234,
    bloodPressure: { systolic: 128, diastolic: 84 },
    oxygenSaturation: 97,
    temperature: 36.9,
    stress: 28
  }
];

const environmentalSensors: EnvironmentalSensor[] = [
  {
    id: 'env-uti-001',
    location: 'UTI Adulto',
    temperature: 22.5,
    humidity: 45,
    airQuality: 95,
    noiseLevel: 42,
    lighting: 350,
    pressure: 1013
  },
  {
    id: 'env-emergencia-002',
    location: 'Emergência',
    temperature: 24.1,
    humidity: 52,
    airQuality: 87,
    noiseLevel: 68,
    lighting: 420,
    pressure: 1011
  },
  {
    id: 'env-cardiologia-003',
    location: 'Cardiologia',
    temperature: 23.2,
    humidity: 48,
    airQuality: 92,
    noiseLevel: 35,
    lighting: 380,
    pressure: 1014
  }
];

export default function IoTIntegration() {
  const [selectedDevice, setSelectedDevice] = useState<IoTDevice | null>(null);
  const [realTimeData, setRealTimeData] = useState(wearableData);
  const [isStreaming, setIsStreaming] = useState(true);

  useEffect(() => {
    if (!isStreaming) return;

    const interval = setInterval(() => {
      setRealTimeData(prev => prev.map(item => ({
        ...item,
        heartRate: Math.max(60, Math.min(100, item.heartRate + (Math.random() - 0.5) * 4)),
        oxygenSaturation: Math.max(95, Math.min(100, item.oxygenSaturation + (Math.random() - 0.5) * 2)),
        temperature: Math.max(36, Math.min(38, item.temperature + (Math.random() - 0.5) * 0.3)),
        stress: Math.max(10, Math.min(50, item.stress + (Math.random() - 0.5) * 8))
      })));
    }, 2000);

    return () => clearInterval(interval);
  }, [isStreaming]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'bg-green-500';
      case 'offline': return 'bg-gray-500';
      case 'warning': return 'bg-yellow-500';
      case 'critical': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getDeviceIcon = (type: string) => {
    switch (type) {
      case 'wearable': return <Watch className="h-5 w-5" />;
      case 'sensor': return <Thermometer className="h-5 w-5" />;
      case 'monitor': return <Monitor className="h-5 w-5" />;
      case 'infusion': return <Activity className="h-5 w-5" />;
      default: return <Router className="h-5 w-5" />;
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <Activity className="h-3 w-3 text-red-500" />;
      case 'down': return <Activity className="h-3 w-3 text-blue-500 rotate-180" />;
      default: return <Activity className="h-3 w-3 text-gray-500" />;
    }
  };

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Integração IoT</h1>
          <p className="text-muted-foreground">
            Monitoramento em tempo real através de dispositivos conectados
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button 
            variant={isStreaming ? "default" : "outline"}
            onClick={() => setIsStreaming(!isStreaming)}
          >
            {isStreaming ? (
              <>
                <Signal className="mr-2 h-4 w-4" />
                Streaming Ativo
              </>
            ) : (
              <>
                <Router className="mr-2 h-4 w-4" />
                Iniciar Stream
              </>
            )}
          </Button>
        </div>
      </div>

      <Tabs defaultValue="devices" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="devices">Dispositivos</TabsTrigger>
          <TabsTrigger value="wearables">Wearables</TabsTrigger>
          <TabsTrigger value="environmental">Ambiente</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="devices" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {iotDevices.map((device) => (
              <Card 
                key={device.id}
                className={`cursor-pointer transition-all hover:shadow-lg ${
                  selectedDevice?.id === device.id ? 'ring-2 ring-blue-500' : ''
                }`}
                onClick={() => setSelectedDevice(device)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      {getDeviceIcon(device.type)}
                      <div className={`h-2 w-2 rounded-full ${getStatusColor(device.status)}`} />
                    </div>
                    <Badge variant={device.status === 'critical' ? 'destructive' : 'secondary'}>
                      {device.status}
                    </Badge>
                  </div>
                  <CardTitle className="text-lg">{device.name}</CardTitle>
                  <CardDescription>{device.location}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Battery className="h-4 w-4" />
                        <span className="text-sm">Bateria</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Progress value={device.batteryLevel} className="w-16 h-2" />
                        <span className="text-sm font-medium">{device.batteryLevel}%</span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Wifi className="h-4 w-4" />
                        <span className="text-sm">Sinal</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Progress value={device.signalStrength} className="w-16 h-2" />
                        <span className="text-sm font-medium">{device.signalStrength}%</span>
                      </div>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Última leitura</span>
                      <span className="font-medium">{device.lastReading}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {selectedDevice && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  {getDeviceIcon(selectedDevice.type)}
                  <span>Detalhes do Dispositivo: {selectedDevice.name}</span>
                </CardTitle>
                <CardDescription>
                  {selectedDevice.location} - {selectedDevice.patientId ? `Paciente: ${selectedDevice.patientId}` : 'Não atribuído'}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-semibold mb-2">Métricas Atuais</h4>
                      <div className="space-y-2">
                        {Object.entries(selectedDevice.metrics).map(([key, metric]) => (
                          <div key={key} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                            <div className="flex items-center space-x-2">
                              {getTrendIcon(metric.trend)}
                              <span className="text-sm capitalize">{key}</span>
                            </div>
                            <div className="flex items-center space-x-2">
                              <span className={`text-sm font-medium ${metric.alert ? 'text-red-600' : ''}`}>
                                {metric.value} {metric.unit}
                              </span>
                              {metric.alert && <AlertTriangle className="h-4 w-4 text-red-500" />}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-2">Status do Sistema</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Status</span>
                          <Badge variant={selectedDevice.status === 'critical' ? 'destructive' : 'secondary'}>
                            {selectedDevice.status}
                          </Badge>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Bateria</span>
                          <span className="font-medium">{selectedDevice.batteryLevel}%</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Força do Sinal</span>
                          <span className="font-medium">{selectedDevice.signalStrength}%</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Última Atualização</span>
                          <span className="font-medium">{selectedDevice.lastReading}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="wearables" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Heart className="h-5 w-5" />
                  <span>Frequência Cardíaca</span>
                </CardTitle>
                <CardDescription>
                  Monitoramento contínuo - Paciente P1847
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={realTimeData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="heartRate" stroke="#ef4444" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Activity className="h-5 w-5" />
                  <span>Saturação de Oxigênio</span>
                </CardTitle>
                <CardDescription>
                  SpO2 - Monitoramento em tempo real
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={realTimeData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <Tooltip />
                    <Area type="monotone" dataKey="oxygenSaturation" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Thermometer className="h-5 w-5" />
                  <span>Temperatura Corporal</span>
                </CardTitle>
                <CardDescription>
                  Monitoramento térmico contínuo
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={realTimeData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="temperature" stroke="#f59e0b" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Gauge className="h-5 w-5" />
                  <span>Nível de Stress</span>
                </CardTitle>
                <CardDescription>
                  Análise de variabilidade cardíaca
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={realTimeData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <Tooltip />
                    <Area type="monotone" dataKey="stress" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.3} />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="environmental" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {environmentalSensors.map((sensor) => (
              <Card key={sensor.id}>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Monitor className="h-5 w-5" />
                    <span>{sensor.location}</span>
                  </CardTitle>
                  <CardDescription>
                    Sensores ambientais em tempo real
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <div className="flex items-center space-x-2">
                        <Thermometer className="h-4 w-4" />
                        <span className="text-sm">Temperatura</span>
                      </div>
                      <span className="font-medium">{sensor.temperature}°C</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <div className="flex items-center space-x-2">
                        <Activity className="h-4 w-4" />
                        <span className="text-sm">Umidade</span>
                      </div>
                      <span className="font-medium">{sensor.humidity}%</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <div className="flex items-center space-x-2">
                        <Gauge className="h-4 w-4" />
                        <span className="text-sm">Qualidade do Ar</span>
                      </div>
                      <span className="font-medium">{sensor.airQuality}%</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <div className="flex items-center space-x-2">
                        <Signal className="h-4 w-4" />
                        <span className="text-sm">Ruído</span>
                      </div>
                      <span className="font-medium">{sensor.noiseLevel} dB</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Dispositivos Conectados</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">Total</span>
                    <span className="font-medium">127</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Online</span>
                    <span className="font-medium text-green-600">119</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Offline</span>
                    <span className="font-medium text-red-600">8</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Alertas Ativos</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">Críticos</span>
                    <span className="font-medium text-red-600">3</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Avisos</span>
                    <span className="font-medium text-yellow-600">12</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Informações</span>
                    <span className="font-medium text-blue-600">7</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Performance</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">Latência Média</span>
                    <span className="font-medium">12ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Uptime</span>
                    <span className="font-medium">99.7%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Dados/hora</span>
                    <span className="font-medium">45.2k</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Alert>
            <CheckCircle className="h-4 w-4" />
            <AlertDescription>
              Todos os sistemas IoT estão operando normalmente. 
              Próxima manutenção preventiva programada para amanhã às 02:00.
            </AlertDescription>
          </Alert>
        </TabsContent>
      </Tabs>
    </div>
  );
}