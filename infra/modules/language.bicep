@description('Azure AI Language resource name')
param name string

@description('Azure region')
param location string

resource language 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: name
  location: location
  kind: 'TextAnalytics'
  sku: { name: 'S' }
  properties: {
    customSubDomainName: name
    publicNetworkAccess: 'Enabled'
  }
}

output endpoint string = language.properties.endpoint
output name string = language.name
