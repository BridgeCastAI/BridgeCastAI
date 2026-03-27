@description('Azure SignalR Service resource name')
param name string

@description('Azure region')
param location string

resource signalR 'Microsoft.SignalRService/signalR@2024-03-01' = {
  name: name
  location: location
  sku: { name: 'Free_F1', capacity: 1 }
  kind: 'SignalR'
  properties: {
    features: [
      { flag: 'ServiceMode', value: 'Serverless' }
    ]
    cors: { allowedOrigins: ['*'] }
    tls: { clientCertEnabled: false }
  }
}

output endpoint string = 'https://${signalR.properties.hostName}'
output name string = signalR.name
