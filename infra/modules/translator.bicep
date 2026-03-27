@description('Azure Translator resource name')
param name string

@description('Azure region (use "global" for Translator)')
param location string

resource translator 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: name
  location: location
  kind: 'TextTranslation'
  sku: { name: 'S1' }
  properties: {
    customSubDomainName: name
    publicNetworkAccess: 'Enabled'
  }
}

output endpoint string = translator.properties.endpoint
output name string = translator.name
