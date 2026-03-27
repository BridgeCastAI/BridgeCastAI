@description('Azure Content Safety resource name')
param name string

@description('Azure region')
param location string

resource contentSafety 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: name
  location: location
  kind: 'ContentSafety'
  sku: { name: 'S0' }
  properties: {
    customSubDomainName: name
    publicNetworkAccess: 'Enabled'
  }
}

output endpoint string = contentSafety.properties.endpoint
output name string = contentSafety.name
