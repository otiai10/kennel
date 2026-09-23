# TLS fixture for the `fetch` tests (test-only)

These files exist only so `tests/providers/test_fetch_http.py` can run an HTTPS server on
loopback and check that `fetch` verifies the certificate against the host name it was given
(`fetch.test`) while connecting to the checked IP address.

- `ca.pem`: a throwaway CA ("kennel test CA"). Its private key was deleted right after signing.
- `fetch-test.pem` / `fetch-test-key.pem`: a server certificate for `DNS:fetch.test`, signed by
  that CA, valid for 100 years.

**The private key is intentionally public.** It protects nothing, and nothing outside the tests
trusts `ca.pem`. Never use these files anywhere else.

They were generated with the system `openssl` (LibreSSL 3.3.6):

```bash
openssl req -x509 -newkey rsa:2048 -nodes -keyout ca-key.pem -out ca.pem -days 36500 \
  -subj "/CN=kennel test CA" -addext "basicConstraints=critical,CA:TRUE" \
  -addext "keyUsage=critical,keyCertSign,cRLSign"
openssl req -newkey rsa:2048 -nodes -keyout fetch-test-key.pem -out leaf.csr -subj "/CN=fetch.test"
printf "subjectAltName=DNS:fetch.test\nbasicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n" > ext.cnf
openssl x509 -req -in leaf.csr -CA ca.pem -CAkey ca-key.pem -CAcreateserial -out fetch-test.pem \
  -days 36500 -extfile ext.cnf
rm leaf.csr ext.cnf ca.srl ca-key.pem
```
