#!/usr/bin/env python3
"""Interface class for all AI clients."""

import logging
import sys
from time import time_ns
from typing import Any, Dict, List

import dynamic_reconfigure.client
import rosnode
import rospy
from actionlib import SimpleActionClient

from neurapy_ai.utils.logging import NeuraLogFormatter


class BaseAiClient:
    """Interface class for all AI clients."""

    def __init__(
        self,
        node_name: str,
        service_proxy: List[rospy.ServiceProxy],
        action_clients: List[SimpleActionClient],
        has_parameters: bool = True,
        log_level: int = 20,
    ):
        """Initialize base client.

        Parameters
        ----------
        node_name : str
            Name of the node
        service_proxy : List[rospy.ServiceProxy]
            List of service proxies that node publishes
        action_clients : List[SimpleActionClient]
            List of action clients that node publishes
        has_parameters : bool, optional
            A flag to tell whether reconfiguration parameters exist for this
            client or not, by default true
        log_level : int, optional
            Set log level for the client, by default 20 (INFO). Options are:
            50 (CRITICAL)
            40 (ERROR)
            30 (WARNING)
            20 (INFO)
            10 (DEBUG)

        Raises
        ----------
        ValueError
            When a given `node_name` is not running
        """
        self._log = logging.getLogger(node_name + "_client")
        # To avoid recognising handlers in parent loggers.
        # This line also causes logs created with this logger to stop
        # getting passed to its parent loggers' handlers.
        self._log.propagate = False

        self._log.setLevel(log_level)
        # Add handler only for new logger objects
        if not self._log.hasHandlers():
            # create formatter and add it to the handlers
            formatter = NeuraLogFormatter()

            # create console handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(log_level)
            ch.setFormatter(formatter)
            self._log.addHandler(ch)

        if not (
            "/" + node_name in rosnode.get_node_names()
            or node_name in rosnode.get_node_names()
        ):
            self._raise_and_log(
                ValueError,
                f"Given node_name {node_name} is not in the list of "
                f"running nodes {rosnode.get_node_names()}!",
            )

        try:
            rospy.get_rostime()
        except rospy.exceptions.ROSInitException:
            # create unique ros node if no node ready
            rospy.init_node(f"neura_ai_client_{time_ns()}")

        for service in service_proxy:
            self._wait_for_service_and_raise(service)

        for action in action_clients:
            self._wait_for_action_and_raise(action)

        self._node_name = node_name

        if has_parameters:
            try:
                self._dyn_client = dynamic_reconfigure.client.Client(
                    node_name, timeout=0.5
                )
                self._default_parameters = self._dyn_client.get_configuration()
                if "groups" in self._default_parameters:
                    del self._default_parameters["groups"]
            except (rospy.ServiceException, rospy.ROSException):
                self._log.debug(
                    "Could not get dynamic parameters for node " + node_name
                )
                self._dyn_client = None

    def _raise_and_log(self, error_type: Exception, message: str):
        """Log the message in ERROR level and raises an exception.

        Parameters
        ----------
        error_type : Exception
            Any exception child class of Python's BaseException class
        message : str
            Error message
        """
        message = "Error: " + message
        self._log.error(message)
        raise error_type(message)

    def _wait_for_service_and_raise(self, service: rospy.ServiceProxy):
        """Check if the service is advertised.

        Parameters
        ----------
        service : rospy.ServiceProxy
            Service proxy of the service that should be checked

        Raises
        ----------
        ConnectionError
            If the service is not advertised
        """
        try:
            service.wait_for_service(5.0)
        except (rospy.ServiceException, rospy.ROSException):
            self._raise_and_log(
                ConnectionError,
                f"Could not connect to service {service.resolved_name}",
            )

    def _wait_for_action_and_raise(self, action_client: SimpleActionClient):
        """Check if the action is advertised.

        Parameters
        ----------
        action_client : SimpleActionClient
            Action client of the action that should be checked

        Raises
        ----------
        ConnectionError
            If the service is not advertised
        """
        if not action_client.wait_for_server(rospy.Duration(2.0)):
            self._raise_and_log(
                ConnectionError,
                "Action " + action_client.action_client.ns + " not available",
            )

    def reset_parameters(self) -> None:
        """Reset this node's parameters to its defaults.

        Raises
        ----------
        SystemError
            If the node has no parameters to reset
        """
        if self._dyn_client is None:
            self._raise_and_log(SystemError, "This node has no parameters")
        self.set_parameters(self._default_parameters)

    def set_parameters(self, params_dict: Dict) -> None:
        """Set multiple parameters of this node from a dictionary.

        Parameters
        ----------
        params_dict : Dict
            Dictionary with parameter names and values to set.

        Raises
        ----------
        SystemError
            If the node has no parameters
        KeyError
            If one of the parameter names in `params_dict` is not a parameter of
            this node
        """
        if self._dyn_client is None:
            self._raise_and_log(SystemError, "This node has no parameters")
        try:
            self._dyn_client.update_configuration(params_dict)
        except dynamic_reconfigure.DynamicReconfigureParameterException:
            self._raise_and_log(
                KeyError,
                f"Could not set parameters {params_dict}  for node "
                + self._node_name,
            )

    def set_parameter(self, param_name: str, value: Any) -> None:
        """Set a single parameter of this node.

        Parameters
        ----------
        param_name : str
            Name of the parameter
        value : Any
            New value for the parameter

        Raises
        ----------
        SystemError
            If the node has no parameters
        KeyError
            If `param_name` is not a parameter of this node
        """
        if self._dyn_client is None:
            self._raise_and_log(SystemError, "This node has no parameters")
        try:
            self._dyn_client.update_configuration({param_name: value})
        except dynamic_reconfigure.DynamicReconfigureParameterException:
            self._raise_and_log(
                KeyError,
                f"Parameter {param_name} does not exist or value {value} "
                f" could not be set for node {self._node_name}",
            )
