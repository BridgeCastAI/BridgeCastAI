@description('Azure AI Vision resource name')
param name string

@description('Azure region')
param location string

resource vision 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: name
  location: location
  kind: 'ComputerVision'
  sku: { name: 'S1' }
  properties: {
    customSubDomainName: name
    publicNetworkAccess: 'Enabled'
  }
}

output endpoint string = vision.properties.endpoint
output name string = vision.name
