# 

```
{
  "issuer": "https://auth0.openai.com/",
  "authorization_endpoint": "https://auth.openai.com/authorize",
  "token_endpoint": "https://auth0.openai.com/oauth/token",
  "device_authorization_endpoint": "https://auth0.openai.com/oauth/device/code",
  "userinfo_endpoint": "https://auth0.openai.com/userinfo",
  "mfa_challenge_endpoint": "https://auth0.openai.com/mfa/challenge",
  "jwks_uri": "https://auth.openai.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth0.openai.com/oidc/register",
  "revocation_endpoint": "https://auth0.openai.com/oauth/revoke",
  "scopes_supported": [
    "openid",
    "profile",
    "offline_access",
    "name",
    "given_name",
    "family_name",
    "nickname",
    "email",
    "email_verified",
    "picture",
    "created_at",
    "identities",
    "phone",
    "address"
  ],
  "response_types_supported": [
    "code",
    "token",
    "id_token",
    "code token",
    "code id_token",
    "token id_token",
    "code token id_token"
  ],
  "code_challenge_methods_supported": ["S256", "plain"],
  "response_modes_supported": ["query", "fragment", "form_post"],
  "subject_types_supported": ["public"],
  "id_token_signing_alg_values_supported": ["HS256", "RS256", "PS256"],
  "token_endpoint_auth_methods_supported": [
    "client_secret_basic",
    "client_secret_post",
    "private_key_jwt"
  ],
  "claims_supported": [
    "aud",
    "auth_time",
    "created_at",
    "email",
    "email_verified",
    "exp",
    "family_name",
    "given_name",
    "iat",
    "identities",
    "iss",
    "name",
    "nickname",
    "phone_number",
    "picture",
    "sub"
  ],
  "request_uri_parameter_supported": false,
  "request_parameter_supported": false,
  "token_endpoint_auth_signing_alg_values_supported": ["RS256", "RS384", "PS256"],
  "backchannel_logout_supported": true,
  "backchannel_logout_session_supported": true
}
```


