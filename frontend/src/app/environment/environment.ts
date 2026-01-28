  export const environment = {
  angUrl: 'http://localhost:4200/home',
  msalConfig: {
    auth: {
      clientId: 'ba222105-643e-4304-88bf-bb8478b3f3cf',
      authority:
        'https://login.microsoftonline.com/0ae51e19-07c8-4e4b-bb6d-648ee58410f4',
    },
  },

  apiConfig: {
    scopes: ['openid', 'email', 'profile'],
    uri: 'api://ba222105-643e-4304-88bf-bb8478b3f3cf/.default',
  },
}