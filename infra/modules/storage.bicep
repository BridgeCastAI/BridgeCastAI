@description('Azure Storage Account name (lowercase, no hyphens)')
param name string

@description('Azure region')
param location string

// Storage account names must be 3-24 chars, lowercase letters and numbers only
var cleanName = toLower(replace(name, '-', ''))
var storageName = length(cleanName) > 24 ? substring(cleanName, 0, 24) : cleanName

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource meetingRecordings 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'meeting-recordings'
  properties: { publicAccess: 'None' }
}

resource meetingPdfs 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'meeting-pdfs'
  properties: { publicAccess: 'None' }
}

resource avatarAssets 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'avatar-assets'
  properties: { publicAccess: 'Blob' }
}

output name string = storage.name
