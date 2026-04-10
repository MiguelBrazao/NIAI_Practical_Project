FROM lscr.io/linuxserver/webtop:ubuntu-xfce

# Set environment variables for the display
ENV PUID=1000
ENV PGID=1000
ENV TZ=Etc/UTC

# Switch to root to install Java
USER root

# Install Java 8 (OpenJDK 8)
# We update apt and install the JDK
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk ant && \
    apt-get clean
WORKDIR /server
# Copy project files
COPY server /server

# 4. Permissions
RUN chown -R 1000:1000 /server

# Copy the runner script and make it executable
COPY run-server.sh /server//run-server.sh
RUN chmod +x /server/run-server.sh

COPY mario.desktop /etc/xdg/autostart/mario.desktop

# Compile
#RUN javac -cp "src:lib/*" -d bin src/ch/idsia/scenarios/Evolve.java

# The start command for this specific image
# It automatically starts the VNC server; we just need to keep the container alive
# You will run the java command manually inside the browser terminal as before.