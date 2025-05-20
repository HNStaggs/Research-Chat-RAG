param location string = resourceGroup().location
param functionAppName string
param storageAccountName string
param appServicePlanName string
param containerName string = 'datasets'

resource storageAccount 'Microsoft.Storage/storageAccounts@2022-09-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
}

resource container 'Microsoft.Storage/storageAccounts/blobServices/containers@2022-09-01' = {
  name: '${storageAccount.name}/default/${containerName}'
  dependsOn: [
    storageAccount
  ]
}

resource appServicePlan 'Microsoft.Web/serverfarms@2022-03-01' = {
  name: appServicePlanName
  location: location
  sku: {
    name: 'Y1'
    tier: 'Dynamic'
  }
}

resource functionApp 'Microsoft.Web/sites@2022-03-01' = {
  name: functionAppName
  location: location
  kind: 'functionapp'
  properties: {
    serverFarmId: appServicePlan.id
    siteConfig: {
      appSettings: [
        {
          name: 'AzureWebJobsStorage'
          value: storageAccount.properties.primaryEndpoints.blob
        }
        {
          name: 'FUNCTIONS_EXTENSION_VERSION'
          value: '~4'
        }
        {
          name: 'FUNCTIONS_WORKER_RUNTIME'
          value: 'python'
        }
        {
          name: 'AZURE_STORAGE_CONNECTION_STRING'
          value: storageAccount.properties.primaryEndpoints.blob
        }
        {
          name: 'AZURE_STORAGE_CONTAINER'
          value: containerName
        }
      ]
    }
  }
  dependsOn: [
    storageAccount
    appServicePlan
    container
  ]
} 