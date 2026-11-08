# syntax=docker/dockerfile:1.7
# The chat UI (D-058): unprivileged nginx serving ui/public and proxying the API allow-list.
# Only the base image pull touches the network, at build time; nothing is fetched at runtime.

ARG NGINX_IMAGE=nginxinc/nginx-unprivileged:1.28-alpine@sha256:7377697a821c131a924a7105fafbe7414db4e9fcc77a6f08f776f33f141ec3f8

FROM ${NGINX_IMAGE}
USER root
RUN rm -rf /usr/share/nginx/html/* /etc/nginx/conf.d/default.conf
COPY ui/public/ /usr/share/nginx/html/
COPY docker/ui-nginx.conf.template /etc/nginx/templates/default.conf.template
RUN chmod -R a+rX,go-w /usr/share/nginx/html /etc/nginx/templates
# Only ^UI_ variables are substituted into the template (nginx's own $variables are kept).
ENV NGINX_ENVSUBST_FILTER=^UI_ \
    UI_UPSTREAM=api:8000
USER 101
EXPOSE 8080
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=10 \
  CMD ["wget", "-q", "-O", "/dev/null", "http://127.0.0.1:8080/healthz"]
