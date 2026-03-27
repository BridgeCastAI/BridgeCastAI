@description('Azure Communication Services resource name')
param name string

@description('Azure region (use "global")')
param location string

resource acs 'Microsoft.Communication/communicationServices@2023-04-01' = {
  name: name
  location: location
  properties: {
    dataLocation: 'United States'
  }
}

output name string = acs.name
