@description('Azure App Service name')
param name string

@description('Azure region')
param location string

resource plan 'Microsoft.Web/serverfarms@2023-12-01' = {
  name: '${name}-plan'
  location: location
  sku: { name: 'B1', tier: 'Basic' }
  kind: 'linux'
  properties: { reserved: true }
}

resource app 'Microsoft.Web/sites@2023-12-01' = {
  name: name
  location: location
  kind: 'app,linux'
  properties: {
    serverFarmId: plan.id
    siteConfig: {
      linuxFxVersion: 'PYTHON|3.11'
      appCommandLine: 'python server/meeting_api.py --host 0.0.0.0 --port 8000'
    }
    httpsOnly: true
  }
}

output name string = app.name
output url string = 'https://${app.properties.defaultHostName}'
